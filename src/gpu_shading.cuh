#pragma once
#include "common.h"
#include "scene.h"
#include "gpu_intersect.cuh" // 包含几何求交 (intersect, trace_shadow, triangle_area)
#include <curand_kernel.h>

// ============================================================================
// 外部资源引用
// ============================================================================
// 纹理句柄数组 (定义在 gpu_context.cu 中)
// 使用 __constant__ 内存以获得最佳缓存性能
#define MAX_TEXTURES 5
__constant__ cudaTextureObject_t d_textures[MAX_TEXTURES];

// ============================================================================
// PBR 物理辅助函数 (BSDF Helpers)
// ============================================================================

// ----------------------------------------------------------------------------
// [Schlick 近似]: 计算菲涅尔反射率 (Fresnel Reflectance)
// ----------------------------------------------------------------------------
// 物理含义: 光线以不同角度打到物体表面时，反射光与折射光的比例。
// 视角越倾斜 (Grazing Angle)，反射越强。
// cosine: 视角与法线夹角 (N dot V)
// F0: 垂直入射时的基础反射率 (Base Reflectivity)
__device__ inline Vec fresnel_schlick(float cosine, Vec F0) {
    // powf(..., 5.0f) 是 Schlick 近似的特征
    return F0 + (make_vec(1.0f,1.0f,1.0f) - F0) * powf(1.0f - cosine, 5.0f);
}

// ----------------------------------------------------------------------------
// [粗糙反射采样]: 生成随机微表面反射方向
// ----------------------------------------------------------------------------
// 物理含义: 模拟粗糙表面。表面越粗糙，反射光线的方向就越发散。
// 我们以"完美反射方向"为中心，根据 roughness 生成一个随机偏移向量。
__device__ inline Vec sample_rough_reflection(Vec perfect_refl, float roughness, curandState* state) {
    float r1 = curand_uniform(state) * 2.0f * M_PI; // 随机角度 0~360
    float r2 = curand_uniform(state);               // 随机半径
    
    // roughness 映射为扰动球的半径
    // roughness = 0 -> radius = 0 -> 完美镜面
    float radius = roughness; 
    
    // 在单位球面上均匀采样
    float z = 1.0f - 2.0f * r2;
    float r = sqrtf(1.0f - z * z);
    
    // 生成扰动向量
    Vec random_sphere = make_vec(r * cosf(r1), r * sinf(r1), z);
    
    // 将扰动加到完美反射方向上，并归一化
    return (perfect_refl + random_sphere * radius).norm();
}

// ============================================================================
// 主渲染内核 (Mega Kernel)
// ============================================================================
// 职责:
// 1. 生成相机光线
// 2. 遍历 BVH 寻找最近交点
// 3. 计算插值属性 (法线, UV)
// 4. 计算 PBR 材质与光照 (NEE)
// 5. 决定下一条光线的方向 (递归模拟)
__global__ void render_kernel_impl(Vec* accum_buffer, int width, int height, int frame_seed, 
                                   CameraParams cam, LinearBVHNode* bvh_nodes, 
                                   Object* scene_objects, int* light_indices, int light_count) {
    
    // ------------------------------------------------------------------------
    // 1. 线程索引与随机数初始化
    // ------------------------------------------------------------------------
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = (height - y - 1) * width + x; // Y轴翻转，匹配屏幕坐标系

    curandState state;
    curand_init(1984 + frame_seed, i, 0, &state);

    // ------------------------------------------------------------------------
    // 2. 相机光线生成 (薄透镜模型 Thin Lens Camera)
    // ------------------------------------------------------------------------
    // 抗锯齿 (Anti-Aliasing): 像素内抖动
    float r1 = 2 * curand_uniform(&state);
    float r2 = 2 * curand_uniform(&state);
    float dx = r1 < 1 ? sqrtf(r1) - 1 : 1 - sqrtf(2 - r1);
    float dy = r2 < 1 ? sqrtf(r2) - 1 : 1 - sqrtf(2 - r2);
    
    // 计算理想针孔方向
    Vec dir_pinhole = (cam.cx * (((x + .5f + dx) / width - .5f)) + 
                       cam.cy * (((y + .5f + dy) / height - .5f)) + cam.dir).norm();
    
    // 景深 (Depth of Field): 透镜采样
    Vec lens_offset = {0, 0, 0};
    if (cam.lens_radius > 0.0f) {
        float lr = cam.lens_radius * sqrtf(curand_uniform(&state));
        float lt = 2 * M_PI * curand_uniform(&state);
        // 在相机平面的基向量 (u,v) 上偏移
        Vec u = cam.cx.norm();
        Vec v = cam.cy.norm();
        lens_offset = u * (lr * cosf(lt)) + v * (lr * sinf(lt));
    }

    // 计算实际光线
    Vec p_focus = cam.pos + dir_pinhole * cam.focus_dist; // 聚焦点
    Vec r_o = cam.pos + lens_offset;                      // 起点 (透镜上某点)
    Vec r_d = (p_focus - r_o).norm();                     // 方向

    // ------------------------------------------------------------------------
    // 3. 路径追踪循环 (Path Tracing Loop)
    // ------------------------------------------------------------------------
    Vec throughput = make_vec(1, 1, 1); // 能量传输比率
    Vec radiance = make_vec(0, 0, 0);   // 累积光能
    const int MAX_DEPTH = 10;           // 最大反弹次数
    const int RR_THRESHOLD = 3;         // 俄罗斯轮盘赌启动阈值
    
    // 记录上一跳类型，防止 NEE 双重计数 (1=SPEC, 2=REFR, 0=DIFF)
    int prev_refl_mode = 1; // 初始设为 SPEC，允许第一帧直接看到发光体

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        
        // --- 3.1 BVH 遍历求交 (Intersection) ---
        
        // 预计算倒数，加速 AABB 检测
        auto safe_inv = [](float x) { return (fabsf(x) < 1e-8f) ? (x >= 0 ? 1e20f : -1e20f) : (1.0f / x); };
        Vec r_inv_d = make_vec(safe_inv(r_d.x), safe_inv(r_d.y), safe_inv(r_d.z));
        
        float d_min = 1e20f;
        int id = -1;
        // 记录重心坐标，用于后续插值
        float hit_u = 0.0f;
        float hit_v = 0.0f;
        
        // 显存栈 (Local Memory Stack)
        int stack[32];
        int ptr = 0;
        stack[ptr++] = 0; // 压入根节点

        while (ptr > 0) {
            int idx = stack[--ptr];
            LinearBVHNode node = bvh_nodes[idx];
            
            // AABB 剔除
            if (!node.bounds.hit(r_o, r_inv_d, 0.0f, d_min)) continue;

            if (node.is_leaf) {
                // 遍历叶子节点内的所有三角形
                for (int k = 0; k < node.primitive_count; k++) {
                    int obj_idx = node.primitive_offset + k;
                    const Object& tri = scene_objects[obj_idx];
                    
                    // [内联 Möller–Trumbore 算法]
                    // 为了获取重心坐标 (u,v)，这里展开写，而不调用 gpu_intersect.cuh 的 intersect
                    // 这样可以避免重复计算
                    Vec e1 = tri.v1 - tri.v0;
                    Vec e2 = tri.v2 - tri.v0;
                    Vec h = r_d.cross(e2);
                    float a = e1.dot(h);
                    
                    if (a > -1e-6f && a < 1e-6f) continue; // 平行
                    
                    float f = 1.0f / a;
                    Vec s = r_o - tri.v0;
                    float u = f * s.dot(h);
                    if (u < 0.0f || u > 1.0f) continue;
                    
                    Vec q = s.cross(e1);
                    float v = f * r_d.dot(q);
                    if (v < 0.0f || u + v > 1.0f) continue;
                    
                    float t = f * e2.dot(q);
                    
                    // 找到更近的交点
                    if (t > 0.0f && t < d_min) {
                        d_min = t;
                        id = obj_idx;
                        hit_u = u;
                        hit_v = v;
                    }
                }
            } else {
                // 内部节点: 压栈子节点
                stack[ptr++] = node.right_child_idx;
                stack[ptr++] = node.left_child_idx;
            }
        }

        // 如果没打中任何物体，结束光路 (背景黑)
        // 进阶: 这里可以采样 HDRI 环境贴图
        if (id < 0) break; 

        // --- 3.2 属性插值 (Interpolation) ---
        const Object& obj = scene_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        float w = 1.0f - hit_u - hit_v; // 重心坐标 w

        // [法线插值]: 实现 Phong Shading (平滑着色)
        Vec n;
        if (obj.use_smooth) {
            // 使用顶点法线插值
            n = obj.vn0 * w + obj.vn1 * hit_u + obj.vn2 * hit_v;
            n.norm(); // 插值后长度可能变短，必须重新归一化
        } else {
            // 使用面法线 (Flat Shading)
            Vec e1 = obj.v1 - obj.v0;
            Vec e2 = obj.v2 - obj.v0;
            n = e1.cross(e2).norm();
        }
        Vec nl = n.dot(r_d) < 0 ? n : n * -1; // 确保法线朝向光线来的一侧

        // [UV 插值]: 实现正确的纹理映射
        // 混合三个顶点的 UV 坐标
        float tex_u = obj.uv0.u * w + obj.uv1.u * hit_u + obj.uv2.u * hit_v;
        float tex_v = obj.uv0.v * w + obj.uv1.v * hit_u + obj.uv2.v * hit_v;

        // --- 3.3 材质准备 ---
        Vec albedo = obj.albedo;
        float metallic = obj.metallic;
        float roughness = obj.roughness;
        float transmission = obj.transmission;

        // [纹理采样]
        if (obj.tex_id >= 0) {
            // 使用插值后的 UV 直接查表
            // tex2D 提供硬件双线性插值
            float4 tex = tex2D<float4>(d_textures[obj.tex_id], tex_u, tex_v);

            // 混合纹理颜色和基础颜色
            albedo = albedo.mult(make_vec(tex.x, tex.y, tex.z));
        }

        // --- 3.4 自发光累加 (Emission) ---
        // 防止 NEE 双重计数:
        // 只有当"直接看灯"或者"从镜面/折射反射过来"时，才累加自发光。
        // 如果是漫反射过来的，因为 NEE 已经算过这盏灯了，所以不加。
        bool is_specular_bounce = (prev_refl_mode == 1) || (prev_refl_mode == 2);
        if (is_specular_bounce) {
             Vec added_light = throughput.mult(obj.emission);
             radiance = radiance + added_light;
        }
        
        // 如果打中强光源，通常光路终止 (除非做透射光)
        if (obj.emission.x > 0.001f || obj.emission.y > 0.001f || obj.emission.z > 0.001f) {
            break; 
        }


        // --- 3.5 俄罗斯轮盘赌 (Russian Roulette) ---
        if (depth > RR_THRESHOLD) {
            float p = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
            if (p < 0.05f) p = 0.05f;
            if (curand_uniform(&state) < p) throughput = throughput * (1.0f / p);
            else break;
        }

        // --- 3.6 PBR 能量权重计算 ---
        // 1. 准备菲涅尔参数
        Vec F0 = make_vec(0.04f, 0.04f, 0.04f);
        F0 = F0 * (1.0f - metallic) + albedo * metallic;
        
        // 使用反向光线计算视角余弦
        float cos_theta = fmaxf(nl.dot(r_d * -1.0f), 0.0f);
        Vec F = fresnel_schlick(cos_theta, F0);
        float F_avg = (F.x + F.y + F.z) / 3.0f;

        // 2. 计算非线性抑制系数
        
        // [漫反射抑制]: 金属度越高，漫反射权重呈指数级下降
        // 0.0 -> 1.0;  0.5 -> 0.125;  1.0 -> 0.0
        float diffuse_scale = powf(1.0f - metallic, 3.0f);

        // [镜面反射抑制]: 粗糙度越高，镜面反射权重下降 (仅针对非金属!)
        // 原始衰减: 1.0 - r^2 (粗糙度1.0时衰减为0)
        float spec_scale = 1.0f - (roughness * roughness);
        if (spec_scale < 0.0f) spec_scale = 0.0f;

        // 3. 计算基础能量权重
        
        // 镜面: 菲涅尔 * 抑制系数
        float w_spec = F_avg * spec_scale;
        
        // 透射: (1-F) * 透射度
        // (玻璃通常比较光滑，暂时不做粗糙度抑制，或者可以复用 spec_scale)
        float w_trans = (1.0f - F_avg) * transmission;
        
        // 漫反射: (1-F) * (1-透射) * 亮度 * 抑制系数
        // 引入 albedo 亮度权重: 如果物体是黑色的，就少采样漫反射
        float albedo_lum = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
        float w_diff = (1.0f - F_avg) * (1.0f - transmission) * albedo_lum * diffuse_scale;

        // 4. 归一化为概率 (PDF)
        float sum = w_spec + w_trans + w_diff;
        
        // 防止除以零 (极暗物体)
        if (sum < 1e-6f) {
            // 默认全给漫反射或者结束
             w_diff = 1.0f; sum = 1.0f;
        }

        float p_spec = w_spec / sum;
        float p_trans = w_trans / sum;
        
        float rnd = curand_uniform(&state);

        // =========================================================
        // 分支 A: 镜面反射 (Specular)
        // =========================================================
        if (rnd < p_spec) {
            Vec perfect = r_d - n * 2 * n.dot(r_d);
            r_d = sample_rough_reflection(perfect, roughness, &state);
            
            // 如果反射进物体内部，视为被吸收
            if (r_d.dot(nl) <= 0.0f) break; 

            // 权重: w_spec / p_spec * F_color_factor
            float weight = 1.0f / p_spec;
            throughput = throughput.mult(F) * weight ; 

            r_o = x_hit + nl * 1e-3f; // 往外推
            prev_refl_mode = 1; // SPEC
        }
        // =========================================================
        // 分支 B: 透射 (Transmission / Glass)
        // =========================================================
        else if (rnd < p_spec + p_trans) {
            bool into = n.dot(nl) > 0;
            float nc = 1.0f, nt = obj.ior;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = r_d.dot(nl);
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
            
            if (cos2t < 0.0f) { 
                // [全内反射 TIR]: 像光纤一样反射回去
                Vec refl = r_d - n * 2.0f * n.dot(r_d);
                r_d = sample_rough_reflection(refl, roughness, &state); // 粗糙内反射
                r_o = x_hit + nl * 1e-3f;
            } else {
                // [折射]: 穿过物体
                Vec tdir = (r_d * nnt - n * ((into ? 1.0f : -1.0f) * (ddn * nnt + sqrtf(cos2t)))).norm();
                
                // 粗糙透射 (Rough Transmission)
                if (roughness > 0.0f) {
                    // 复用 sample_rough_reflection 的逻辑，但以透射方向为轴
                    float r1 = curand_uniform(&state) * 2.0f * M_PI;
                    float r2 = curand_uniform(&state);
                    float radius = roughness; 
                    float z = 1.0f - 2.0f * r2;
                    float r = sqrtf(1.0f - z * z);
                    Vec random_sphere = make_vec(r * cosf(r1), r * sinf(r1), z);
                    tdir = (tdir + random_sphere * radius).norm();
                }
                
                r_d = tdir; 
                r_o = x_hit - nl * 1e-3f; // 往里推 (穿过去)
            }
            
            float weight = 1.0f / p_trans;
            throughput = throughput.mult(albedo) * weight;
            prev_refl_mode = 2; // REFR
        }
        // =========================================================
        // 分支 C: 漫反射 (Diffuse + NEE)
        // =========================================================
        else {
            
            // --- NEE (直接光照采样) ---
            if (light_count > 0 ) {
                // 1. 随机选灯
                int l_idx = (int)(curand_uniform(&state) * (light_count - 0.001f));
                const Object& light = scene_objects[light_indices[l_idx]];
                
                // 2. 在灯上选点 (三角形均匀采样)
                float r1 = curand_uniform(&state); float r2 = curand_uniform(&state);
                float sqr1 = sqrtf(r1);
                float u = 1.0f - sqr1; float v = sqr1 * (1.0f - r2);
                Vec light_pos = light.v0 * u + light.v1 * v + light.v2 * (1.0f - u - v);
                
                // 3. 几何因子
                Vec to_light = light_pos - x_hit;
                float dist_sq = to_light.dot(to_light);
                float dist = sqrtf(dist_sq);
                Vec L_dir = to_light * (1.0f / dist);
                
                float cos_theta = nl.dot(L_dir);
                if (cos_theta > 0.0f) {
                     // 灯的法线
                     Vec le1 = light.v1 - light.v0; Vec le2 = light.v2 - light.v0;
                     Vec ln = le1.cross(le2).norm();
                     float cos_light = -ln.dot(L_dir);

                     if (cos_light > 0.0f) {
                         // 4. 阴影测试 (trace_shadow)
                         if (!trace_shadow(x_hit + nl*1e-3f, L_dir, dist-1e-2f, bvh_nodes, scene_objects)) {
                             float area = triangle_area(light);
                             float pdf = 1.0f / (area * light_count);
                             float G = (cos_theta * cos_light) / dist_sq;
                             Vec brdf = albedo * (1.0f / M_PI);
                             Vec contrib = light.emission.mult(brdf) * (G / pdf);
                             
                             // 累加直接光 (别忘了乘 throughput)
                             radiance = radiance + throughput.mult(contrib);
                         }
                     }
                }
            }

            // --- 漫反射反弹 (采样下一跳方向) ---
            float r1 = 2 * M_PI * curand_uniform(&state);
            float r2 = curand_uniform(&state);
            float r2s = sqrtf(r2);
            Vec w = nl;
            Vec temp = (fabs(w.x) > 0.1f ? make_vec(0, 1, 0) : make_vec(1, 0, 0));
            Vec u_vec = temp.cross(w).norm();
            Vec v_vec = w.cross(u_vec);
            r_d = (u_vec * cosf(r1) * r2s + v_vec * sinf(r1) * r2s + w * sqrtf(1 - r2)).norm();
            
            float p_diff_real = 1.0f - p_spec - p_trans;
            float weight = 1.0f / p_diff_real;
            throughput = throughput.mult(albedo) * weight;
            r_o = x_hit + nl * 1e-3f;
            prev_refl_mode = 0; // DIFF
        }
    }
    
    // ------------------------------------------------------------------------
    // 5. 写入结果 (Output)
    // ------------------------------------------------------------------------
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) ||
        isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
        return; // 直接丢弃这次采样
    }
    
    // 智能亮度压缩 (Despeckle)
    float final_lum = radiance.x * 0.2 + radiance.y * 0.7 + radiance.z * 0.1;
    if (final_lum > 100.0f) radiance = radiance * (20.0f / final_lum);
    
    accum_buffer[i] = accum_buffer[i] + radiance;
}
