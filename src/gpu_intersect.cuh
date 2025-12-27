#pragma once

#include "common.h"
#include "scene.h"
#include "bvh.h"

// ======================================================================================
// 几何计算核心模块 (Device Code)
// ======================================================================================
// 职责:
// 1. 计算三角形面积 (用于采样 PDF)。
// 2. 射线-三角形求交 (Möller–Trumbore 算法)。
// 3. 阴影遮挡测试 (BVH 遍历 + 快速提前退出)。
// ======================================================================================

// --------------------------------------------------------------------------------------
// [辅助] 计算三角形面积
// --------------------------------------------------------------------------------------
// 面积 = 0.5 * |AB x AC|
__device__ inline float triangle_area(const Object& obj) {
    Vec e1 = obj.v1 - obj.v0;
    Vec e2 = obj.v2 - obj.v0;
    Vec c = e1.cross(e2);
    // 长度函数: sqrt(x*x + y*y + z*z)
    float len = sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);
    return len * 0.5f;
}

// --------------------------------------------------------------------------------------
// [核心] 射线-三角形求交 (Möller–Trumbore Algorithm)
// --------------------------------------------------------------------------------------
// 这是一个极度优化的算法，它不需要计算三角形所在的平面方程，
// 而是直接通过基底变换求解重心坐标 (u, v) 和 距离 (t)。
//
// 参数:
//   obj: 三角形数据 (必须包含 v0, v1, v2)
//   r_o: 射线起点
//   r_d: 射线方向
//
// 返回:
//   float: 交点距离 t。如果未相交或在背面，返回 0.0。
// --------------------------------------------------------------------------------------
__device__ inline float intersect(const Object& obj, const Vec& r_o, const Vec& r_d) {
    const float eps = 1e-6f; // 数值稳定性阈值
    
    // 1. 计算两条边向量
    Vec e1 = obj.v1 - obj.v0;
    Vec e2 = obj.v2 - obj.v0;
    
    // 2. 计算行列式 (Determinant) 相关参数
    // h = r_d X e2
    Vec h = r_d.cross(e2);
    float a = e1.dot(h);
    
    // [平行检测]
    // 如果 a 接近 0，说明光线平行于三角形平面 -> 没打中
    if (a > -eps && a < eps) return 0.0f;
    
    float f = 1.0f / a;
    Vec s = r_o - obj.v0;
    
    // 3. 计算重心坐标 u
    // u 必须在 [0, 1] 之间
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return 0.0f;
    
    // 4. 计算重心坐标 v
    // v 必须在 [0, 1] 之间，且 u + v <= 1
    Vec q = s.cross(e1);
    float v = f * r_d.dot(q);
    if (v < 0.0f || u + v > 1.0f) return 0.0f;
    
    // 5. 计算距离 t
    float t = f * e2.dot(q);
    
    // [有效性检查]
    // t 必须大于 eps (防止打中光线起点的那个面，即 Self-Intersection)
    if (t > eps) {
        return t;
    } else {
        return 0.0f;
    }
}

// --------------------------------------------------------------------------------------
// [核心] 阴影光线测试 (Shadow Ray / Visibility Test)
// --------------------------------------------------------------------------------------
// 专门用于 NEE (直接光照) 的检测。
// 
// 优化策略:
// 1. "Any Hit": 不需要找最近的交点，只要找到*任意*一个挡在中间的物体，立刻返回 true。
// 2. 提前退出: 一旦由 true，循环直接终止，不再遍历剩下的 BVH 树。
//
// 参数:
//   nodes: BVH 节点数组指针 (Global Memory)
//   scene_objects: 物体数组指针 (Global Memory)
// --------------------------------------------------------------------------------------
__device__ inline bool trace_shadow(const Vec& origin, const Vec& dir, float max_dist, 
                                    LinearBVHNode* nodes, Object* scene_objects) {
    
    // 预计算光线方向倒数，加速 AABB 检测
    // 这里的 1e-8f 保护是为了防止除以 0 产生 Inf
    auto safe_inv = [](float x) { 
        return (fabsf(x) < 1e-8f) ? (x >= 0 ? 1e20f : -1e20f) : (1.0f / x); 
    };
    Vec r_inv_d = make_vec(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    
    // 显存栈 (Local Memory Stack)
    // 深度 32 对于平衡的 BVH 树足够容纳数亿个三角形
    int stack[32];
    int ptr = 0;
    stack[ptr++] = 0; // 压入根节点
                      
    // [新增] 安全计数器
    int safety_counter = 0;
    const int MAX_STEPS = 200; // 阴影测试通常很快，200次遍历足够了


    while (ptr > 0 && safety_counter++ < MAX_STEPS) {
        // 弹出节点
        int idx = stack[--ptr];
        LinearBVHNode node = nodes[idx];

        // [AABB 测试]
        // 使用 max_dist 作为 t_max。
        // 如果盒子虽然被光线穿过，但在光源之后(> max_dist)，则不算遮挡，直接跳过。
        if (!node.bounds.hit(origin, r_inv_d, 0.001f, max_dist)) {
            continue;
        }

        // [叶子节点]
        if (node.primitive_count > 0) { // count > 0 表示是叶子 (根据 bvh.h 定义)
            for (int k = 0; k < node.primitive_count; k++) {
                int obj_idx = node.primitive_offset + k;
                
                // 对具体的三角形求交
                float t = intersect(scene_objects[obj_idx], origin, dir);
                
                // [关键]: 只要发现阻挡，立即返回 true
                // t > 0.001: 防止自我遮挡 (打中出发点)
                // t < max_dist - 0.001: 防止打中光源本身 (如果是面光源，终点就是光源上的点)
                if (t > 0.001f && t < max_dist - 0.001f) {
                    return true; // 被挡住了 (Shadowed)
                }
            }
        } 
        // [内部节点]
        else {
            // 将左右子节点压栈
            // 简单的压栈顺序。进阶优化可以根据光线方向决定先压哪个，
            // 但对于 Shadow Ray (Any Hit)，顺序优化的收益不如 Primary Ray 大。
            stack[ptr++] = node.right_child_idx;
            stack[ptr++] = node.left_child_idx;
        }
    }
    
    return false; // 一路通畅，没有遮挡 (Lit)
}
