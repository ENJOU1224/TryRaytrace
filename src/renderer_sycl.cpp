#include "renderer.h"
#include "common.h"
#include "aabb.h"
#include "bvh.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

/**
 * ======================================================================================
 * [SYCL 渲染后端实现] 
 * 针对 Intel Xe2 架构 (Lunar Lake) 进行硬件指令级优化
 * ======================================================================================
 */

// -----------------------------------------------------------------------------
// [全局变量] 存储在设备可见的 USM 内存中
// -----------------------------------------------------------------------------
static queue* g_queue = nullptr;
static Object* d_objects = nullptr;
static LinearBVHNode* d_bvh_nodes = nullptr;
static int* d_light_indices = nullptr;
static int d_light_count = 0;

/**
 * @brief 高性能 Hash 随机数生成器 (PCG 算法简化版)
 * 用于生成像素内采样和 BSDF 反射所需的白噪声
 */
struct Random {
    unsigned int state;
    HOST_DEVICE Random(unsigned int pixel_idx, unsigned int seed) {
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

/**
 * @brief 路径追踪主算法 (Path Tracing Core)
 * 采用迭代式路径追踪，支持 BVH 遍历、PBR 材质、NEE 显式光源采样和 RR 俄罗斯轮盘赌
 */
HOST_DEVICE Vec trace(Vec r_o, Vec r_d, Random& rng, const LinearBVHNode* bvh_nodes, const Object* scene_objects, const int* light_indices, int l_count) {
    Vec radiance = {0, 0, 0};
    Vec throughput = {1, 1, 1};
    const int MAX_DEPTH = 10;
    bool prev_was_specular = true; // 默认 true 以捕捉直接入眼的光

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // [1] BVH 树遍历：查找场景中最近交点
        float d_min = 1e20f;
        int id = -1;
        int stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; 
        
        Vec r_inv_d = {1.0f/r_d.x, 1.0f/r_d.y, 1.0f/r_d.z};

        while (stack_ptr > 0) {
            int idx = stack[--stack_ptr];
            const auto& node = bvh_nodes[idx];
            
            // AABB 包围盒求交 (硬件级 Slab Method)
            if (!node.bounds.hit(r_o, r_inv_d, 0.001f, d_min)) continue;
            
            if (node.is_leaf) {
                for (int k = 0; k < node.primitive_count; k++) {
                    int obj_idx = node.primitive_offset + k;
                    // 射线-三角形求交
                    Vec e1 = scene_objects[obj_idx].v1 - scene_objects[obj_idx].v0;
                    Vec e2 = scene_objects[obj_idx].v2 - scene_objects[obj_idx].v0;
                    Vec h = r_d.cross(e2); 
                    float a = e1.dot(h);
                    if (a > -1e-5f && a < 1e-5f) continue;
                    float f = 1.0f / a;
                    Vec s = r_o - scene_objects[obj_idx].v0;
                    float u = f * s.dot(h);
                    if (u < 0.0f || u > 1.0f) continue;
                    Vec q = s.cross(e1);
                    float v = f * r_d.dot(q);
                    if (v < 0.0f || u + v > 1.0f) continue;
                    float t = f * e2.dot(q);
                    if (t > 0.001f && t < d_min) { d_min = t; id = obj_idx; }
                }
            } else {
                stack[stack_ptr++] = node.right_child_idx;
                stack[stack_ptr++] = node.left_child_idx;
            }
        }

        // 光线逸出场景
        if (id < 0) break;

        const Object& obj = scene_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        Vec n = (obj.v1 - obj.v0).cross(obj.v2 - obj.v0).norm();
        Vec nl = n.dot(r_d) < 0 ? n : n * -1; // 修正后的法线方向

        // [2] 累加自发光 (仅当满足镜面标记或第一帧，防止 NEE 重复计数)
        if (prev_was_specular) {
            radiance = radiance + throughput.mult(obj.emission);
        }
        if (obj.emission.norm_len() > 0.1f) break;

        // [3] PBR 材质参数预备
        Vec albedo = obj.albedo;
        float metallic = obj.metallic, roughness = obj.roughness, transmission = obj.transmission;
        float cos_theta = std::abs(r_d.dot(nl));

        // 计算 Fresnel 项
        Vec f0 = albedo * metallic + Vec{0.04f, 0.04f, 0.04f} * (1.0f - metallic);
        Vec F = fresnel_schlick(cos_theta, f0);
        float p_spec = (F.x + F.y + F.z) * 0.3333f;

        float rnd = rng.next_float();
        
        // [4] BSDF 采样分支选择 (重要性采样)
        if (rnd < transmission) {
            // --- 折射分支 (Glass) ---
            float ior = obj.ior > 0 ? obj.ior : 1.5f;
            bool into = n.dot(nl) > 0; float nnt = into ? 1.0f / ior : ior;
            float ddn = r_d.dot(nl); float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
            if (cos2t < 0) r_d = (r_d - n * 2.0f * r_d.dot(n)).norm();
            else r_d = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + std::sqrt(cos2t)))).norm();
            r_o = x_hit + r_d * 0.001f; 
            throughput = throughput.mult(albedo) * (1.0f / std::max(transmission, 0.01f));
            prev_was_specular = true;
        } 
        else if (rnd < transmission + p_spec || (roughness < 0.03f)) {
            // --- 镜面反射分支 (Metal/Mirror) ---
            r_d = (r_d - n * 2.0f * r_d.dot(n)).norm();
            // 粗糙表面扰动
            if (roughness > 0.03f) {
                float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float();
                float r = std::sqrt(std::max(0.0f, 1.0f - r2 * r2));
                Vec rand_v = {r * std::cos(r1), r * std::sin(r1), r2};
                r_d = (r_d + rand_v * roughness).norm();
            }
            if(r_d.dot(nl) < 0) break;
            r_o = x_hit + nl * 0.001f; 
            // 确定性镜面判定：如果足够光滑，执行无损能量传输以消除噪声
            float weight = (roughness < 0.03f) ? 1.0f : std::max(p_spec, 0.01f);
            throughput = throughput.mult(F) * (1.0f / weight);
            prev_was_specular = true;
        } 
        else {
            // --- 漫反射分支 (Diffuse + NEE) ---
            // NEE: 直接采样光源
            if (l_count > 0 && depth < 3) {
                const Object& light = scene_objects[light_indices[(int)(rng.next_float() * l_count)]];
                float lu = 1.0f - std::sqrt(rng.next_float());
                float lv = rng.next_float() * (1.0f - lu);
                Vec lp = light.v0 * lu + light.v1 * lv + light.v2 * (1.0f - lu - lv);
                Vec tl = lp - x_hit;
                float dist_sq = std::max(tl.dot(tl), 0.01f);
                float dist = std::sqrt(dist_sq);
                Vec ld = tl * (1.0f / dist);

                if (nl.dot(ld) > 0) {
                    bool blocked = false; int s_ptr = 0, s_stack[32]; s_stack[s_ptr++] = 0;
                    Vec s_inv = {1.0f/ld.x, 1.0f/ld.y, 1.0f/ld.z};
                    while(s_ptr > 0) {
                        int idx = s_stack[--s_ptr];
                        if(!bvh_nodes[idx].bounds.hit(x_hit + nl * 0.001f, s_inv, 0.001f, dist - 0.01f)) continue;
                        if(bvh_nodes[idx].is_leaf) {
                            for(int k=0; k<bvh_nodes[idx].primitive_count; k++) {
                                int s_obj_idx = bvh_nodes[idx].primitive_offset+k;
                                Vec se1 = scene_objects[s_obj_idx].v1 - scene_objects[s_obj_idx].v0;
                                Vec se2 = scene_objects[s_obj_idx].v2 - scene_objects[s_obj_idx].v0;
                                Vec sh = ld.cross(se2); float sa = se1.dot(sh);
                                if (sa > -1e-5f && sa < 1e-5f) continue;
                                float sf = 1.0f / sa; Vec ss = (x_hit + nl * 0.001f) - scene_objects[s_obj_idx].v0;
                                float su = sf * ss.dot(sh); if (su < 0.0f || su > 1.0f) continue;
                                Vec sq = ss.cross(se1); float sv = sf * ld.dot(sq);
                                if (sv < 0.0f || su + sv > 1.0f) continue;
                                // 阴影测试只应统计“光源之前”的遮挡物。
                                // 如果不加上界判断，被采样到的灯三角形本身也可能被误判为遮挡，
                                // 直接光会大面积失效，路径追踪只能依赖随机命中光源，方差会急剧上升。
                                float st = sf * se2.dot(sq);
                                if (st > 0.001f && st < dist - 0.01f) { blocked = true; break; }
                            }
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

            // 漫反射随机采样 (Lambertian)
            float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float(), r2s = std::sqrt(r2);
            Vec w = nl, basis_u = (((w.x > 0 ? w.x : -w.x) > 0.1f ? Vec{0,1,0} : Vec{1,0,0}).cross(w)).norm(), basis_v = w.cross(basis_u);
            r_d = (basis_u * std::cos(r1) * r2s + basis_v * std::sin(r1) * r2s + w * std::sqrt(std::max(0.0f, 1.0f-r2))).norm();
            r_o = x_hit + nl * 0.001f; 
            float p_diff = std::max(1.0f - transmission - p_spec, 0.01f);
            throughput = throughput.mult(albedo) * (1.0f / p_diff);
            prev_was_specular = false; 
        }

        // [5] 俄罗斯轮盘赌 (RR)
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
// [接口实现] 
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
    auto objects = d_objects; auto nodes = d_bvh_nodes; auto lights = d_light_indices; int l_count = d_light_count;
    g_queue->submit([&](handler& h) {
        h.parallel_for(nd_range<2>(range<2>(width, height), range<2>(tx, ty)), [=](nd_item<2> item) {
            int x = item.get_global_id(0); int y = item.get_global_id(1);
            if (x >= width || y >= height) return;
            int i = y * width + x; 
            Random rng(i, frame_seed);
            
            // 抗锯齿抖动
            float fx = (float)(x + rng.next_float() - 0.5f) / width - 0.5f;
            float fy = 0.5f - (float)(y + rng.next_float() - 0.5f) / height;
            
            Vec r_d = (cam.cx * fx + cam.cy * fy + cam.dir).norm();
            Vec color = trace(cam.pos, r_d, rng, nodes, objects, lights, l_count);
            
            // --- 数值稳定性防火墙 ---
            if (std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z) || std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) color = {0,0,0};
            color.x = std::max(0.0f, color.x); color.y = std::max(0.0f, color.y); color.z = std::max(0.0f, color.z);
            
            // Firefly Clamping (严格亮度钳制，治理异常噪点)
            float lum = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
            if (lum > 5.0f) color = color * (5.0f / lum); 
            
            accum_buffer_usm[i] = accum_buffer_usm[i] + color;
        });
    });
    g_queue->wait();
}
