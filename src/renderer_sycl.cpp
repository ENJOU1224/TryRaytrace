#include "renderer.h"
#include "common.h"
#include "aabb.h"
#include "bvh.h"
#include "ray_query_backend.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

/**
 * ======================================================================================
 * [SYCL 渲染后端实现] 
 * 针对 Intel Xe2 架构 (Lunar Lake) 进行硬件指令级优化
 *
 * 这个文件是项目里“最像渲染器内核”的地方。
 * 如果你第一次读路径追踪代码，建议按下面顺序看：
 * 1. 先看最底部的 launch_render_kernel()，理解“一像素一线程”
 * 2. 再看 trace()，理解“一条光线如何一路反弹”
 * 3. 然后回头看 find_closest_hit() / is_shadowed() / sample_light()
 * 4. 最后再看上面的统计和 clamp 辅助函数
 * ======================================================================================
 */

// -----------------------------------------------------------------------------
// [全局变量] 存储在设备可见的 USM 内存中
// -----------------------------------------------------------------------------
static queue* g_queue = nullptr;
// d_objects / d_bvh_nodes / d_light_indices 都是“设备可见的场景副本”。
// 主程序初始化场景后，会把 CPU 上的数据复制到这些 USM 指针里。
static Object* d_objects = nullptr;
static LinearBVHNode* d_bvh_nodes = nullptr;
static int* d_light_indices = nullptr;
static int d_light_count = 0;

// 下面这些常量大多是“工程上为了稳住噪点”的经验参数。
// 它们不是严格物理公式的一部分，而是为了让当前项目在实时反馈下更稳定。
constexpr float kDirectLightLumLimit = 3.0f;
constexpr float kEmissiveHitLumLimit = 4.0f;
constexpr float kThroughputMaxComponent = 4.0f;
constexpr float kFinalFireflyLumLimit = 2.0f;
constexpr bool kApplyFinalFireflyClamp = true;
// 射线起点偏移一个很小量，避免新射线立刻和自己刚命中的面再次相交。
constexpr float kRayEpsilon = 0.001f;
// 路径最多反弹多少次。
constexpr int kMaxPathDepth = 5;
// 从第几次 bounce 开始允许俄罗斯轮盘赌提前终止。
constexpr int kRussianRouletteStartDepth = 3;
// NEE 只在前几层做，后面层数太深时收益会下降。
constexpr int kMaxNeeDepth = 3;

struct LightSample {
    // 采样点指向光源点的单位方向
    Vec direction = {0.0f, 0.0f, 0.0f};
    // hit_point 到采样光点的距离
    float distance = 0.0f;
    // 距离平方，常用于光照公式里的 1 / r^2
    float distance_sq = 0.0f;
    // 光源三角形面积
    float area = 0.0f;
    // 光源法线与 -light_dir 的夹角余弦
    float light_cos = 0.0f;
    // 指回被采样到的灯对象，方便读取 emission
    const Object* light = nullptr;
    // 有些采样在几何上无效（例如背面），这时直接标 false
    bool valid = false;
};

/**
 * @brief 高性能 Hash 随机数生成器 (PCG 算法简化版)
 * 用于生成像素内采样和 BSDF 反射所需的白噪声
 */
struct Random {
    unsigned int state;
    HOST_DEVICE Random(unsigned int pixel_idx, unsigned int seed) {
        // 把“像素坐标 + 帧种子”混成一个足够乱的初始状态。
        // 这样不同像素、不同帧都能得到不同随机序列。
        unsigned int h = pixel_idx * 0x45d9f3bu + seed;
        h = ((h >> 16u) ^ h) * 0x45d9f3bu;
        h = ((h >> 16u) ^ h) * 0x45d9f3bu;
        state = (h >> 16u) ^ h;
    }
    HOST_DEVICE unsigned int next() {
        unsigned int oldstate = state;
        state = oldstate * 747796405u + 2891336453u;
        unsigned int word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
        return (word >> 22u) ^ word;
    }
    HOST_DEVICE float next_float() {
        // 1 / 2^32，把 uint32 映射到 [0, 1) 浮点区间。
        return (float)next() * 2.3283064365386963e-10f; 
    }
};

/**
 * @brief Schlick 菲涅尔近似公式
 * 计算视角相关的反射率权重
 */
HOST_DEVICE Vec fresnel_schlick(float cos_theta, Vec f0) {
    float x = 1.0f - cos_theta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return f0 + (Vec{1,1,1} - f0) * x5;
}

HOST_DEVICE float luminance(const Vec& color) {
    // 使用标准 Rec.709 权重估算亮度。
    // 后面的 firefly clamp 和统计都依赖它。
    return color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
}

HOST_DEVICE float max_component(const Vec& color) {
    // 对 throughput 来说，最大通道值是一个很实用的“能量是否失控”代理量。
    return fmax_wrapper(color.x, fmax_wrapper(color.y, color.z));
}

HOST_DEVICE uint32_t encode_milli(float value) {
    // 把浮点值编码成 milli 整数，是为了在 GPU 端做原子 max 更简单。
    float safe = value < 0.0f ? 0.0f : value;
    return static_cast<uint32_t>(safe * 1000.0f + 0.5f);
}

template <bool EnableStats>
HOST_DEVICE void stats_increment(uint32_t* counter) {
    if constexpr (EnableStats) {
        // 这里用 atomic_ref，是因为统计缓冲区放在全局 USM 里，多个 work-item 会同时写。
        sycl::atomic_ref<uint32_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> atomic_counter(*counter);
        atomic_counter.fetch_add(1);
    }
}

template <bool EnableStats>
HOST_DEVICE void stats_update_max(uint32_t* counter, float value) {
    if constexpr (EnableStats) {
        // 统计最大值时同样使用原子 max。
        // 这不是为了业务逻辑，而是为了后续离线分析“最极端的样本到底有多亮”。
        sycl::atomic_ref<uint32_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> atomic_counter(*counter);
        atomic_counter.fetch_max(encode_milli(value));
    }
}

template <bool EnableStats>
HOST_DEVICE Vec clamp_direct_light(Vec contribution, RenderStats* stats) {
    // 直接光一般应该比间接高亮更稳定。
    // 这里的 clamp 更像“把极端离群样本拉回可控范围”，不是常规 tone mapping。
    const float lum = luminance(contribution);
    stats_update_max<EnableStats>(&stats->max_direct_light_lum_milli, lum);
    if (lum > kDirectLightLumLimit) {
        stats_increment<EnableStats>(&stats->direct_light_clamp_count);
        contribution = contribution * (kDirectLightLumLimit / lum);
    }
    return contribution;
}

template <bool EnableStats>
HOST_DEVICE Vec clamp_emissive_hit(Vec contribution, RenderStats* stats, bool primary_hit) {
    // 这类贡献对应“路径最终命中了发光体”。
    // 由于它可能是非常高能的低概率事件，所以单独统计、单独限幅。
    const float lum = luminance(contribution);
    stats_update_max<EnableStats>(&stats->max_emissive_hit_lum_milli, lum);
    if constexpr (EnableStats) {
        if (primary_hit) {
            stats_increment<true>(&stats->emissive_hit_primary_count);
            stats_update_max<true>(&stats->max_emissive_hit_primary_lum_milli, lum);
        } else {
            stats_increment<true>(&stats->emissive_hit_indirect_count);
            stats_update_max<true>(&stats->max_emissive_hit_indirect_lum_milli, lum);
        }
    }
    if (lum > kEmissiveHitLumLimit) {
        if constexpr (EnableStats) {
            stats_increment<true>(&stats->emissive_hit_clamp_count);
            if (primary_hit) {
                stats_increment<true>(&stats->emissive_hit_primary_clamp_count);
            } else {
                stats_increment<true>(&stats->emissive_hit_indirect_clamp_count);
            }
        }
        contribution = contribution * (kEmissiveHitLumLimit / lum);
    }
    return contribution;
}

template <bool EnableStats>
HOST_DEVICE Vec clamp_throughput(Vec throughput, uint32_t* clamp_counter, RenderStats* stats) {
    // throughput 是整条路径的累计权重。
    // 一旦它某个通道飙太高，后面哪怕只是普通的发光体，也会变成巨亮离群样本。
    const float max_comp = max_component(throughput);
    stats_update_max<EnableStats>(&stats->max_throughput_component_milli, max_comp);
    if (max_comp > kThroughputMaxComponent) {
        if constexpr (EnableStats) {
            if (clamp_counter) {
                stats_increment<true>(clamp_counter);
            }
        }
        throughput = throughput * (kThroughputMaxComponent / max_comp);
    }
    return throughput;
}

/**
 * @brief 从一个三角形面光源上采样一点
 */
HOST_DEVICE LightSample sample_light(const Object& light, const Vec& hit_point, const Vec& surface_normal, Random& rng) {
    LightSample sample;
    // 这里对三角形面光源做重心坐标采样。
    // 这种写法简单，而且对三角形均匀采样是正确的。
    // 直觉上可以理解成：先随机挑出一个“面积均匀”的三角形内部点。
    float lu = 1.0f - std::sqrt(rng.next_float());
    float lv = rng.next_float() * (1.0f - lu);
    Vec light_v1 = light.v0 + light.edge1;
    Vec light_v2 = light.v0 + light.edge2;
    Vec light_point = light.v0 * lu + light_v1 * lv + light_v2 * (1.0f - lu - lv);
    Vec to_light = light_point - hit_point;
    float distance_sq = std::max(to_light.dot(to_light), 0.01f);
    float distance = std::sqrt(distance_sq);
    Vec light_dir = to_light * (1.0f / distance);

    if (surface_normal.dot(light_dir) <= 0.0f) {
        // 表面背向光源时，这次直接光采样没有意义，直接返回 invalid。
        return sample;
    }

    if (light.area <= 0.0f) {
        return sample;
    }

    float light_cos = light.normal.dot(light_dir * -1.0f);
    if (light_cos < 0.0f) {
        light_cos = -light_cos;
    }

    sample.direction = light_dir;
    sample.distance = distance;
    sample.distance_sq = distance_sq;
    sample.area = light.area;
    sample.light_cos = light_cos;
    sample.light = &light;
    sample.valid = true;
    return sample;
}

/**
 * @brief 路径追踪主算法 (Path Tracing Core)
 * 采用迭代式路径追踪，支持 BVH 遍历、PBR 材质、NEE 显式光源采样和 RR 俄罗斯轮盘赌
 */
template <bool EnableStats>
HOST_DEVICE Vec trace(Vec r_o,
                      Vec r_d,
                      Random& rng,
                      const ray_query_backend::SceneView& scene,
                      RenderStats* stats) {
    // 可以把这整个函数理解成：
    // “从相机发出一条光线，不断命中物体、不断改方向、不断累计贡献，
    //  直到跑出场景或被提前终止。”

    // radiance: 当前路径已经累积到的出射辐射
    Vec radiance = {0, 0, 0};
    // throughput: 当前路径从相机走到这里的累计权重
    Vec throughput = {1, 1, 1};
    // prev_was_specular: 上一次 bounce 是否属于镜面类路径
    // 作用是避免 NEE 与“命中灯”重复计数
    bool prev_was_specular = true;

    for (int depth = 0; depth < kMaxPathDepth; depth++) {
        // 1. 找最近交点
        ray_query_backend::SceneHit hit =
            ray_query_backend::find_closest_hit(scene, r_o, r_d, kRayEpsilon);

        // 光线逸出场景，整条路径到此结束
        if (hit.object_id < 0) break;

        const Object& obj = scene.objects[hit.object_id];
        Vec x_hit = r_o + r_d * hit.t;
        Vec n = obj.normal;
        // nl 是“面向来射光线的法线”
        // 路径追踪里通常会用这个方向去构造半球采样和偏移起点。
        Vec nl = n.dot(r_d) < 0 ? n : n * -1;
        bool is_emissive = obj.emission.norm_len() > 0.1f;

        // 2. 累加自发光
        // 只有前一跳是镜面类路径时，才允许这里直接把灯记入 radiance。
        // 否则 diffuse 分支已经通过 NEE 估计过直接光，再记一次会双计。
        if (prev_was_specular && is_emissive) {
            Vec emissive_hit = throughput.mult(obj.emission);
            emissive_hit = clamp_emissive_hit<EnableStats>(emissive_hit, stats, depth == 0);
            radiance = radiance + emissive_hit;
        }

        // 真命中灯后，路径可以直接结束。
        if (is_emissive) break;

        // 3. 准备材质参数
        // 这里把“材质会怎么反弹光线”简化成三个大类：
        // - transmission: 折射
        // - p_spec: 镜面/金属
        // - 剩下的概率: 漫反射
        Vec albedo = obj.albedo;
        float metallic = obj.metallic, roughness = obj.roughness, transmission = obj.transmission;
        float cos_theta = std::abs(r_d.dot(nl));
        bool is_rough_dielectric = (metallic < 0.1f) && (transmission < 0.01f) && (roughness > 0.5f);

        // 计算 Fresnel 项
        Vec f0 = albedo * metallic + Vec{0.04f, 0.04f, 0.04f} * (1.0f - metallic);
        Vec F = fresnel_schlick(cos_theta, f0);

        // 诊断结论：
        // 纯 Cornell Box 墙面本应近似理想漫反射，但原逻辑里即便是 metallic=0、roughness=1 的墙，
        // 仍会因为 dielectric Fresnel 获得约 4% 的 specular 分支概率。
        // 这会让“看不到灯、也没有茶壶”的视角里，依旧存在不少间接命中灯的高能路径。
        // 为了验证这一点，这里先让“高粗糙非金属非透射材质”退回纯漫反射。
        float p_spec = is_rough_dielectric ? 0.0f : (F.x + F.y + F.z) * 0.3333f;

        // 用一个随机数决定本次路径落入哪种 BSDF 分支。
        // 这就是 Monte Carlo 采样的核心味道：
        // 不是“三种都算”，而是“按概率选一种，再用权重补偿”。
        float rnd = rng.next_float();
        
        // [4] BSDF 采样分支选择 (重要性采样)
        if (rnd < transmission) {
            // --- 折射分支 (Glass) ---
            // 这里是简化版介质折射，没有走完整的微表面折射模型。
            float ior = obj.ior > 0 ? obj.ior : 1.5f;
            bool into = n.dot(nl) > 0; float nnt = into ? 1.0f / ior : ior;
            float ddn = r_d.dot(nl); float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
            if (cos2t < 0) r_d = (r_d - n * 2.0f * r_d.dot(n)).norm();
            else r_d = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + std::sqrt(cos2t)))).norm();
            r_o = x_hit + r_d * kRayEpsilon;
            // 因为本次只采样了“折射”这一支，所以要除以它被采中的概率，
            // 才能保持整体估计无偏。
            throughput = throughput.mult(albedo) * (1.0f / std::max(transmission, 0.01f));
            throughput = clamp_throughput<EnableStats>(throughput, EnableStats ? &stats->throughput_clamp_refract_count : nullptr, stats);
            prev_was_specular = true;
        } 
        else if (rnd < transmission + p_spec || (roughness < 0.03f)) {
            // --- 镜面反射分支 (Metal/Mirror) ---
            r_d = (r_d - n * 2.0f * r_d.dot(n)).norm();
            // 粗糙表面扰动：
            // 当前项目没有上完整 GGX 采样，而是用“完美反射 + 随机扰动”的工程化简版。
            if (roughness > 0.03f) {
                float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float();
                float r = std::sqrt(std::max(0.0f, 1.0f - r2 * r2));
                Vec rand_v = {r * std::cos(r1), r * std::sin(r1), r2};
                r_d = (r_d + rand_v * roughness).norm();
            }
            if(r_d.dot(nl) < 0) break;
            r_o = x_hit + nl * kRayEpsilon;
            // 如果足够光滑，就把它近似成确定性镜面，减少不必要噪声。
            float weight = (roughness < 0.03f) ? 1.0f : std::max(p_spec, 0.01f);
            // F 是菲涅尔反射率，表示“这一跳有多少能量被反射出来”。
            throughput = throughput.mult(F) * (1.0f / weight);
            throughput = clamp_throughput<EnableStats>(throughput, EnableStats ? &stats->throughput_clamp_specular_count : nullptr, stats);
            prev_was_specular = true;
        } 
        else {
            // --- 漫反射分支 (Diffuse + NEE) ---
            // 这里会分两件事：
            // 1. 先做一次显式光源采样（NEE）
            // 2. 再做一次半球随机采样，继续往场景里追踪
            if (scene.light_count > 0 && depth < kMaxNeeDepth) {
                const Object& light =
                    scene.objects[scene.light_indices[(int)(rng.next_float() * scene.light_count)]];
                LightSample light_sample = sample_light(light, x_hit, nl, rng);
                if (light_sample.valid &&
                    !ray_query_backend::is_shadowed(scene,
                                                    x_hit + nl * kRayEpsilon,
                                                    light_sample.direction,
                                                    kRayEpsilon,
                                                    light_sample.distance)) {
                    // 直接光的估计值。
                    // 这里本质还是 path tracing，只是把“灯的那一段”显式采样出来，
                    // 方差比纯靠随机命中灯要小很多。
                    Vec direct_light = throughput.mult(light.emission.mult(albedo)) *
                                       (nl.dot(light_sample.direction) * light_sample.light_cos * light_sample.area /
                                        (light_sample.distance_sq * M_PI * (1.0f / scene.light_count)));
                    direct_light = clamp_direct_light<EnableStats>(direct_light, stats);
                    radiance = radiance + direct_light;
                }
            }

            // 漫反射随机采样 (Lambertian)
            // 这是整条路径真正继续往场景里走的那一步。
            float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float(), r2s = std::sqrt(r2);
            Vec w = nl, basis_u = (((w.x > 0 ? w.x : -w.x) > 0.1f ? Vec{0,1,0} : Vec{1,0,0}).cross(w)).norm(), basis_v = w.cross(basis_u);
            r_d = (basis_u * std::cos(r1) * r2s + basis_v * std::sin(r1) * r2s + w * std::sqrt(std::max(0.0f, 1.0f-r2))).norm();
            r_o = x_hit + nl * kRayEpsilon;
            float p_diff = std::max(1.0f - transmission - p_spec, 0.01f);
            // Lambert 漫反射这里同样要除以“本分支被选中的概率”。
            throughput = throughput.mult(albedo) * (1.0f / p_diff);
            throughput = clamp_throughput<EnableStats>(throughput, EnableStats ? &stats->throughput_clamp_diffuse_count : nullptr, stats);
            prev_was_specular = false; 
        }

        // 5. 俄罗斯轮盘赌 (RR)
        // 当路径够深时，用概率方式提前终止长路径，避免把时间都耗在贡献很小的尾部 bounce 上。
        if (depth >= kRussianRouletteStartDepth) {
            // 这里用 albedo 最大通道当作“还有没有必要继续追”的粗略代理。
            // 值越小，说明这条路径后面剩余贡献大概率已经很弱。
            float p = albedo.x > albedo.y ? (albedo.x > albedo.z ? albedo.x : albedo.z) : (albedo.y > albedo.z ? albedo.y : albedo.z);
            if (p < 0.1f) p = 0.1f;
            if (rng.next_float() > p) break;
            throughput = throughput * (1.0f / p);
            throughput = clamp_throughput<EnableStats>(throughput, EnableStats ? &stats->throughput_clamp_rr_count : nullptr, stats);
        }
    }
    return radiance;
}

// -----------------------------------------------------------------------------
// [接口实现] 
// -----------------------------------------------------------------------------

void init_renderer_sycl() {
    if (!g_queue) {
        try { 
            // 优先选择 GPU；如果当前环境不支持，再回退到 CPU。
            g_queue = new queue(gpu_selector_v); 
            std::cout << "[SYCL] Device: " << g_queue->get_device().get_info<info::device::name>() << std::endl;
            std::cout << "[RayQuery] Backend: " << ray_query_backend::kSelectedBackendName << std::endl;
        } catch (...) { 
            g_queue = new queue(cpu_selector_v); 
            std::cout << "[SYCL] Fallback to CPU." << std::endl;
            std::cout << "[RayQuery] Backend: " << ray_query_backend::kSelectedBackendName << std::endl;
        }
    }
}

queue& get_renderer_queue() {
    init_renderer_sycl();
    return *g_queue;
}

void init_scene_data(const std::vector<Object>& objects, const std::vector<std::string>& texture_files, const std::vector<LinearBVHNode>& nodes, const std::vector<int>& light_indices) {
    init_renderer_sycl();
    (void)texture_files;
    // 这里做的是“CPU 场景 -> 设备可见 USM 缓冲”的一次性复制。
    // 场景不变时，这些数据之后每一帧都可以反复复用。
    ray_query_backend::shutdown_scene_backend(*g_queue);
    if (d_objects) free(d_objects, *g_queue);
    d_objects = malloc_shared<Object>(objects.size(), *g_queue);
    std::memcpy(d_objects, objects.data(), objects.size() * sizeof(Object));
    if (d_bvh_nodes) free(d_bvh_nodes, *g_queue);
    d_bvh_nodes = malloc_shared<LinearBVHNode>(nodes.size(), *g_queue);
    std::memcpy(d_bvh_nodes, nodes.data(), nodes.size() * sizeof(LinearBVHNode));
    d_light_count = light_indices.size();
    if (d_light_indices) free(d_light_indices, *g_queue);
    d_light_indices = nullptr;
    if (d_light_count > 0) {
        d_light_indices = malloc_shared<int>(d_light_count, *g_queue);
        std::memcpy(d_light_indices, light_indices.data(), d_light_count * sizeof(int));
    }

    ray_query_backend::initialize_scene_backend(*g_queue, objects, nodes);
}

void launch_render_kernel(Vec* accum_buffer_usm,
                          int width,
                          int height,
                          int frame_seed,
                          int tx,
                          int ty,
                          CameraParams cam,
                          RenderStats* stats_usm) {
    // 把全局设备指针拷贝到局部变量中，便于 lambda 捕获。
    auto objects = d_objects; auto nodes = d_bvh_nodes; auto lights = d_light_indices; int l_count = d_light_count;
    ray_query_backend::SceneView scene = {objects, nodes, lights, l_count};
    ray_query_backend::set_scene_view_backend_state(scene);
    if (stats_usm) {
        g_queue->submit([&](handler& h) {
            h.parallel_for(nd_range<2>(range<2>(width, height), range<2>(tx, ty)), [=](nd_item<2> item) {
                // 一 个 work-item 负责一个像素。
                int x = item.get_global_id(0); int y = item.get_global_id(1);
                if (x >= width || y >= height) return;
                int i = y * width + x;
                Random rng(i, frame_seed);

                // 给像素中心加一点随机抖动，等价于做最基础的抗锯齿 / 随机超采样。
                float fx = (float)(x + rng.next_float() - 0.5f) / width - 0.5f;
                float fy = 0.5f - (float)(y + rng.next_float() - 0.5f) / height;

                // 用相机参数把屏幕像素映射成一条世界空间射线。
                Vec r_d = (cam.cx * fx + cam.cy * fy + cam.dir).norm();
                Vec color = trace<true>(cam.pos, r_d, rng, scene, stats_usm);

                // 数值兜底：如果出现 NaN/Inf，就把这个像素记黑并做统计。
                if (std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z) || std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) {
                    stats_increment<true>(&stats_usm->nan_or_inf_pixels);
                    color = {0, 0, 0};
                }
                color.x = std::max(0.0f, color.x); color.y = std::max(0.0f, color.y); color.z = std::max(0.0f, color.z);

                float lum = luminance(color);
                stats_update_max<true>(&stats_usm->max_final_color_lum_milli, lum);
                if (lum > kFinalFireflyLumLimit) {
                    stats_increment<true>(&stats_usm->final_firefly_clamp_count);
                    if (kApplyFinalFireflyClamp) {
                        color = color * (kFinalFireflyLumLimit / lum);
                    }
                }

                // 这里不是覆盖，而是累加。
                // 因为路径追踪的最终显示结果依赖“多帧平均”。
                accum_buffer_usm[i] = accum_buffer_usm[i] + color;
            });
        });
    } else {
        g_queue->submit([&](handler& h) {
            h.parallel_for(nd_range<2>(range<2>(width, height), range<2>(tx, ty)), [=](nd_item<2> item) {
                int x = item.get_global_id(0); int y = item.get_global_id(1);
                if (x >= width || y >= height) return;
                int i = y * width + x;
                Random rng(i, frame_seed);

                float fx = (float)(x + rng.next_float() - 0.5f) / width - 0.5f;
                float fy = 0.5f - (float)(y + rng.next_float() - 0.5f) / height;

                Vec r_d = (cam.cx * fx + cam.cy * fy + cam.dir).norm();
                Vec color = trace<false>(cam.pos, r_d, rng, scene, nullptr);

                if (std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z) || std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) {
                    color = {0, 0, 0};
                }
                color.x = std::max(0.0f, color.x); color.y = std::max(0.0f, color.y); color.z = std::max(0.0f, color.z);

                float lum = luminance(color);
                if (lum > kFinalFireflyLumLimit && kApplyFinalFireflyClamp) {
                    color = color * (kFinalFireflyLumLimit / lum);
                }

                accum_buffer_usm[i] = accum_buffer_usm[i] + color;
            });
        });
    }
    // main.cpp 设计成“发一帧就等这一帧完成”，所以这里直接 wait。
    // 如果以后要做更深的 CPU/GPU 异步流水线，这里会是一个可继续优化的点。
    g_queue->wait();
}
