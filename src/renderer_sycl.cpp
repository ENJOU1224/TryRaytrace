#include "renderer.h"
#include "common.h"
#include "aabb.h"
#include "bvh.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

/*
 * ======================================================================================
 * [SYCL 渲染后端] 针对 Intel Lunar Lake (Xe2) 架构深度优化
 * ======================================================================================
 * 
 * 核心技术栈:
 * 1. USM (Unified Shared Memory): 实现 CPU 与 GPU 之间的零拷贝数据交换。
 * 2. 硬件加速指令: 使用 sycl::native 指令集加速平方根、三角函数等运算。
 * 3. 数值稳定性: 引入了多重噪声治理 (Firefly Clamping, NaN/Inf 防火墙)。
 */

// -----------------------------------------------------------------------------
// 1. 全局资源管理
// -----------------------------------------------------------------------------
static queue* g_queue = nullptr;
static Object* d_objects = nullptr;
static LinearBVHNode* d_bvh_nodes = nullptr;
static int* d_light_indices = nullptr;
static int d_light_count = 0;

// [高强度随机数生成器] 基于 PCG 算法
// 比简单的 LCG 随机性更强，能有效减少采样模式产生的规律性噪点
struct Random {
    unsigned int state;
    HOST_DEVICE Random(unsigned int pixel_idx, unsigned int seed) {
        // 使用两个不同的 Hash 混合，确保相邻像素的序列完全正交
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
        return (float)next() * 2.3283064365386963e-10f; 
    }
};

// -----------------------------------------------------------------------------
// 2. 数学与求交核心 (Pure C++)
// -----------------------------------------------------------------------------

// [Schlick 菲涅尔近似] 
HOST_DEVICE Vec fresnel_schlick(float cos_theta, Vec f0) {
    float x = 1.0f - cos_theta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return f0 + (Vec{1,1,1} - f0) * x5;
}

// [射线-三角形求交] 使用 Möller–Trumbore 算法
HOST_DEVICE float intersect_triangle(const Object& obj, const Vec& r_o, const Vec& r_d) {
    const float eps = 1e-5f;
    Vec e1 = obj.v1 - obj.v0;
    Vec e2 = obj.v2 - obj.v0;
    Vec h = r_d.cross(e2); 
    float a = e1.dot(h);
    
    if (a > -eps && a < eps) return 0.0f; // 平行
    
    float f = 1.0f / a; 
    Vec s = r_o - obj.v0;
    float u = f * s.dot(h); 
    if (u < 0.0f || u > 1.0f) return 0.0f;
    
    Vec q = s.cross(e1); 
    float v = f * r_d.dot(q);
    if (v < 0.0f || u + v > 1.0f) return 0.0f;
    
    float t = f * e2.dot(q); 
    return (t > eps) ? t : 0.0f;
}

// -----------------------------------------------------------------------------
// 3. 路径追踪主算法 (The Heart)
// -----------------------------------------------------------------------------
HOST_DEVICE Vec trace(Vec r_o, Vec r_d, Random& rng, const LinearBVHNode* bvh_nodes, const Object* scene_objects, const int* light_indices, int l_count) {
    Vec radiance = {0, 0, 0};
    Vec throughput = {1, 1, 1};
    const int MAX_DEPTH = 10;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // [A] BVH 加速遍历 (Stack-based)
        float d_min = 1e20f;
        int id = -1;
        int stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // 根节点入栈
        
        Vec r_inv_d = {1.0f/r_d.x, 1.0f/r_d.y, 1.0f/r_d.z};

        while (stack_ptr > 0) {
            int idx = stack[--stack_ptr];
            const auto& node = bvh_nodes[idx];
            
            // AABB 测试
            if (!node.bounds.hit(r_o, r_inv_d, 0.001f, d_min)) continue;
            
            if (node.is_leaf) {
                for (int k = 0; k < node.primitive_count; k++) {
                    int obj_idx = node.primitive_offset + k;
                    float t = intersect_triangle(scene_objects[obj_idx], r_o, r_d);
                    if (t > 0 && t < d_min) { d_min = t; id = obj_idx; }
                }
            } else {
                // 简单的遍历顺序优化：先入右，再入左，确保先访问左子树
                stack[stack_ptr++] = node.right_child_idx;
                stack[stack_ptr++] = node.left_child_idx;
            }
        }

        // 没打中任何物体
        if (id < 0) break;

        const Object& obj = scene_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        Vec n = (obj.v1 - obj.v0).cross(obj.v2 - obj.v0).norm();
        Vec nl = n.dot(r_d) < 0 ? n : n * -1; // 修正后的几何法线

        // [B] 自发光累加
        radiance = radiance + throughput.mult(obj.emission);
        if (obj.emission.norm_len() > 0.1f) break; // 击中光源，结束光路

        // [C] 材质属性获取
        Vec albedo = obj.albedo;
        float metallic = obj.metallic;
        float roughness = obj.roughness;
        float transmission = obj.transmission;
        float cos_theta = (r_d.dot(nl) < 0) ? -r_d.dot(nl) : r_d.dot(nl);

        // 计算菲涅尔项
        Vec f0 = albedo * metallic + Vec{0.04f, 0.04f, 0.04f} * (1.0f - metallic);
        Vec F = fresnel_schlick(cos_theta, f0);
        float p_spec = (F.x + F.y + F.z) * 0.3333f;

        float rnd = rng.next_float();

        // -----------------------------------------------------
        // [D] BSDF 分支选择 (Importance Sampling)
        // -----------------------------------------------------
        if (rnd < transmission) {
            // [分支 1]: 折射 (Refraction)
            float ior = obj.ior > 0 ? obj.ior : 1.5f;
            bool into = n.dot(nl) > 0;
            float nnt = into ? 1.0f / ior : ior;
            float ddn = r_d.dot(nl);
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

            if (cos2t < 0) { // 全反射
                r_d = (r_d - n * 2.0f * r_d.dot(n)).norm();
            } else {
                r_d = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + std::sqrt(cos2t)))).norm();
            }
            r_o = x_hit + r_d * 0.001f; 
            float weight = (transmission > 0.01f) ? transmission : 0.01f;
            throughput = throughput.mult(albedo) * (1.0f / weight);
        } 
        else if (rnd < transmission + p_spec) {
            // [分支 2]: 镜面反射 (Specular)
            Vec perfect = (r_d - n * 2.0f * r_d.dot(n)).norm();
            float r1 = 2 * M_PI * rng.next_float();
            float r2 = rng.next_float();
            float r = std::sqrt(std::max(0.0f, 1.0f - r2 * r2));
            Vec rand_v = {r * std::cos(r1), r * std::sin(r1), r2};
            
            // 应用粗糙度扰动
            r_d = (perfect + rand_v * roughness).norm();
            if(r_d.dot(nl) < 0) break; // 射入表面内部，废弃光路
            
            r_o = x_hit + nl * 0.001f; 
            float weight = (p_spec > 0.01f) ? p_spec : 0.01f;
            throughput = throughput.mult(F) * (1.0f / weight);
        } 
        else {
            // [分支 3]: 漫反射 (Diffuse + NEE)
            // [NEE]: 显式采样光源，大幅提升收敛速度
            if (l_count > 0 && depth < 3) {
                const Object& light = scene_objects[light_indices[(int)(rng.next_float() * l_count)]];
                float lu = 1.0f - std::sqrt(rng.next_float());
                float lv = rng.next_float() * (1.0f - lu);
                Vec lp = light.v0 * lu + light.v1 * lv + light.v2 * (1.0f - lu - lv);
                Vec tl = lp - x_hit;
                float dist_sq = tl.dot(tl);
                float dist = std::sqrt(dist_sq);
                Vec ld = tl * (1.0f / dist);

                if (nl.dot(ld) > 0) {
                    // 阴影测试 (Shadow Ray)
                    bool blocked = false;
                    int s_ptr = 0, s_stack[32]; s_stack[s_ptr++] = 0;
                    Vec s_inv = {1.0f/ld.x, 1.0f/ld.y, 1.0f/ld.z};
                    while(s_ptr > 0) {
                        int idx = s_stack[--s_ptr];
                        if(!bvh_nodes[idx].bounds.hit(x_hit + nl * 0.001f, s_inv, 0.001f, dist - 0.01f)) continue;
                        if(bvh_nodes[idx].is_leaf) {
                            for(int k=0; k<bvh_nodes[idx].primitive_count; k++)
                                if(intersect_triangle(scene_objects[bvh_nodes[idx].primitive_offset+k], x_hit + nl * 0.001f, ld) > 0) { blocked = true; break; }
                            if(blocked) break;
                        } else { s_stack[s_ptr++] = bvh_nodes[idx].right_child_idx; s_stack[s_ptr++] = bvh_nodes[idx].left_child_idx; }
                    }
                    
                    if (!blocked) {
                        float area = (light.v1 - light.v0).cross(light.v2 - light.v0).norm_len() * 0.5f;
                        float dot_ln = ((light.v1 - light.v0).cross(light.v2 - light.v0).norm()).dot(ld * -1.0f);
                        float cos_l = (dot_ln < 0) ? -dot_ln : dot_ln;
                        radiance = radiance + throughput.mult(light.emission.mult(albedo)) * (nl.dot(ld) * cos_l * area / (dist_sq * M_PI * (1.0f/l_count)));
                    }
                }
            }

            // 漫反射采样 (Lambertian)
            float r1 = 2 * M_PI * rng.next_float();
            float r2 = rng.next_float();
            float r2s = std::sqrt(r2);
            Vec w = nl;
            Vec basis_u = (( (w.x > 0 ? w.x : -w.x) > 0.1f ? Vec{0,1,0} : Vec{1,0,0}).cross(w)).norm();
            Vec basis_v = w.cross(basis_u);
            r_d = (basis_u * std::cos(r1) * r2s + basis_v * std::sin(r1) * r2s + w * std::sqrt(std::max(0.0f, 1.0f-r2))).norm();
            r_o = x_hit + nl * 0.001f; 
            
            float p_diff = 1.0f - transmission - p_spec;
            float weight = (p_diff > 0.01f) ? p_diff : 0.01f;
            throughput = throughput.mult(albedo) * (1.0f / weight);
        }

        // [E] 俄罗斯轮盘赌 (Russian Roulette)
        // 只有在路径深度增加后才启用，防止过早结束重要路径
        if (depth > 3) {
            float p = albedo.x > albedo.y ? (albedo.x > albedo.z ? albedo.x : albedo.z) : (albedo.y > albedo.z ? albedo.y : albedo.z);
            if (p < 0.1f) p = 0.1f;
            if (rng.next_float() > p) break;
            throughput = throughput * (1.0f / p);
        }
    }
    return radiance;
}

// -----------------------------------------------------------------------------
// 4. 外部调用接口
// -----------------------------------------------------------------------------

void init_renderer_sycl() {
    if (!g_queue) {
        try { 
            g_queue = new queue(gpu_selector_v); 
            std::cout << "[SYCL] Device: " << g_queue->get_device().get_info<info::device::name>() << std::endl;
        } catch (...) { 
            g_queue = new queue(cpu_selector_v); 
            std::cout << "[SYCL] Fallback to CPU." << std::endl;
        }
    }
}

void init_scene_data(const std::vector<Object>& objects, const std::vector<std::string>& texture_files, const std::vector<LinearBVHNode>& nodes, const std::vector<int>& light_indices) {
    init_renderer_sycl();
    
    // 分配 USM Shared 内存 (CPU/GPU 通用)
    if (d_objects) free(d_objects, *g_queue);
    d_objects = malloc_shared<Object>(objects.size(), *g_queue);
    std::memcpy(d_objects, objects.data(), objects.size() * sizeof(Object));

    if (d_bvh_nodes) free(d_bvh_nodes, *g_queue);
    d_bvh_nodes = malloc_shared<LinearBVHNode>(nodes.size(), *g_queue);
    std::memcpy(d_bvh_nodes, nodes.data(), nodes.size() * sizeof(LinearBVHNode));

    d_light_count = light_indices.size();
    if (d_light_indices) free(d_light_indices, *g_queue);
    if (d_light_count > 0) {
        d_light_indices = malloc_shared<int>(d_light_count, *g_queue);
        std::memcpy(d_light_indices, light_indices.data(), d_light_count * sizeof(int));
    }
}

void launch_render_kernel(Vec* accum_buffer_usm, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    auto objects = d_objects; 
    auto nodes = d_bvh_nodes; 
    auto lights = d_light_indices; 
    int l_count = d_light_count;

    g_queue->submit([&](handler& h) {
        h.parallel_for(nd_range<2>(range<2>(width, height), range<2>(tx, ty)), [=](nd_item<2> item) {
            int x = item.get_global_id(0);
            int y = item.get_global_id(1);
            if (x >= width || y >= height) return;

            int i = y * width + x; 
            Random rng(i, frame_seed);

            // 抗锯齿抖动采样
            float fx = (float)(x + rng.next_float() - 0.5f) / width - 0.5f;
            float fy = 0.5f - (float)(y + rng.next_float() - 0.5f) / height;
            
            Vec r_d = (cam.cx * fx + cam.cy * fy + cam.dir).norm();
            Vec color = trace(cam.pos, r_d, rng, nodes, objects, lights, l_count);

            // --- 鲁棒性防火墙 ---
            // 1. NaN/Inf 过滤
            if (std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z) || 
                std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) {
                color = {0, 0, 0};
            }
            // 2. 负值清理
            color.x = std::max(0.0f, color.x); 
            color.y = std::max(0.0f, color.y); 
            color.z = std::max(0.0f, color.z);

            // 3. Firefly Clamping (严格限制)
            float lum = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
            if (lum > 5.0f) color = color * (5.0f / lum); 

            accum_buffer_usm[i] = accum_buffer_usm[i] + color;
        });
    });
    g_queue->wait(); // 硬件同步
}
