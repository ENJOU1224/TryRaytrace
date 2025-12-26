#pragma once
#include "common.h"

// 兼容 CPU/GPU 的 min/max 函数
__host__ __device__ inline float fmin_wrapper(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float fmax_wrapper(float a, float b) { return a > b ? a : b; }

// ======================================================================================
// AABB (轴对齐包围盒)
// ======================================================================================
struct __align__(16) AABB {
    Vec min;
    Vec max;

    // [初始化]: 生成一个"无效"的盒子 (min=无穷大, max=无穷小)
    // 这样任何点放进去都会让它变成有效盒子
    __host__ __device__ static AABB empty() {
        return { 
            {1e30f, 1e30f, 1e30f}, 
            {-1e30f, -1e30f, -1e30f} 
        };
    }

    // [扩展]: 把一个点包进来
    __host__ __device__ void grow(Vec p) {
        min.x = fmin_wrapper(min.x, p.x);
        min.y = fmin_wrapper(min.y, p.y);
        min.z = fmin_wrapper(min.z, p.z);

        max.x = fmax_wrapper(max.x, p.x);
        max.y = fmax_wrapper(max.y, p.y);
        max.z = fmax_wrapper(max.z, p.z);
    }

    // [扩展]: 把另一个盒子包进来
    __host__ __device__ void grow(const AABB& other) {
        min.x = fmin_wrapper(min.x, other.min.x);
        min.y = fmin_wrapper(min.y, other.min.y);
        min.z = fmin_wrapper(min.z, other.min.z);

        max.x = fmax_wrapper(max.x, other.max.x);
        max.y = fmax_wrapper(max.y, other.max.y);
        max.z = fmax_wrapper(max.z, other.max.z);
    }

    // [核心算法]: Slab Method 光线求交
    // 返回: 是否相交
    // t_min, t_max: 光线有效的距离范围 (例如 0.001 ~ 1e20)
    __device__ bool hit(const Vec& r_o, const Vec& r_inv_d, float t_min, float t_max) const {
        // X 轴
        float tx1 = (min.x - r_o.x) * r_inv_d.x;
        float tx2 = (max.x - r_o.x) * r_inv_d.x;
        float tmin = fmin_wrapper(tx1, tx2);
        float tmax = fmax_wrapper(tx1, tx2);

        // Y 轴
        float ty1 = (min.y - r_o.y) * r_inv_d.y;
        float ty2 = (max.y - r_o.y) * r_inv_d.y;
        tmin = fmax_wrapper(tmin, fmin_wrapper(ty1, ty2));
        tmax = fmin_wrapper(tmax, fmax_wrapper(ty1, ty2));

        // Z 轴
        float tz1 = (min.z - r_o.z) * r_inv_d.z;
        float tz2 = (max.z - r_o.z) * r_inv_d.z;
        tmin = fmax_wrapper(tmin, fmin_wrapper(tz1, tz2));
        tmax = fmin_wrapper(tmax, fmax_wrapper(tz1, tz2));

        return tmax >= tmin && tmax > t_min && tmin < t_max;
    }
};
