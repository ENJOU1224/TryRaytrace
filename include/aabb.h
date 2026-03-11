#pragma once
#include "common.h"

// AABB = Axis-Aligned Bounding Box，中文常叫“轴对齐包围盒”。
// 它是 BVH 里最重要的基础积木：
// - 一个包围盒负责包住若干三角形
// - 射线先和盒子做便宜的相交测试
// - 只有“可能打中盒子”时，才继续测里面真正的三角形
//
// 这就是 BVH 能加速的关键：先粗筛，再细算。

// 兼容 CPU/GPU 的 min/max 函数
HOST_DEVICE inline float fmin_wrapper(float a, float b) { return a < b ? a : b; }
HOST_DEVICE inline float fmax_wrapper(float a, float b) { return a > b ? a : b; }

// ======================================================================================
// AABB (轴对齐包围盒)
// ======================================================================================
struct ALIGN(16) AABB {
    // min / max 分别表示包围盒在三个坐标轴上的最小点和最大点。
    // 因为盒子始终和坐标轴平行，所以只需要这两个角点就能完整描述它。
    Vec min;
    Vec max;

    HOST_DEVICE static AABB empty() {
        return { 
            {1e30f, 1e30f, 1e30f}, 
            {-1e30f, -1e30f, -1e30f} 
        };
    }

    HOST_DEVICE void grow(Vec p) {
        min.x = fmin_wrapper(min.x, p.x);
        min.y = fmin_wrapper(min.y, p.y);
        min.z = fmin_wrapper(min.z, p.z);

        max.x = fmax_wrapper(max.x, p.x);
        max.y = fmax_wrapper(max.y, p.y);
        max.z = fmax_wrapper(max.z, p.z);
    }

    HOST_DEVICE void grow(const AABB& other) {
        min.x = fmin_wrapper(min.x, other.min.x);
        min.y = fmin_wrapper(min.y, other.min.y);
        min.z = fmin_wrapper(min.z, other.min.z);

        max.x = fmax_wrapper(max.x, other.max.x);
        max.y = fmax_wrapper(max.y, other.max.y);
        max.z = fmax_wrapper(max.z, other.max.z);
    }

    HOST_DEVICE bool hit(const Vec& r_o, const Vec& r_inv_d, float t_min, float t_max) const {
        // slab test 思想：
        // 分别计算射线穿过 x / y / z 三组平行平面的区间，
        // 最后取三个区间的重叠部分。
        // 只要三个轴上的有效区间有交集，就说明射线穿过了盒子。
        float tx1 = (min.x - r_o.x) * r_inv_d.x;
        float tx2 = (max.x - r_o.x) * r_inv_d.x;
        float tmin = fmin_wrapper(tx1, tx2);
        float tmax = fmax_wrapper(tx1, tx2);

        float ty1 = (min.y - r_o.y) * r_inv_d.y;
        float ty2 = (max.y - r_o.y) * r_inv_d.y;
        tmin = fmax_wrapper(tmin, fmin_wrapper(ty1, ty2));
        tmax = fmin_wrapper(tmax, fmax_wrapper(ty1, ty2));

        float tz1 = (min.z - r_o.z) * r_inv_d.z;
        float tz2 = (max.z - r_o.z) * r_inv_d.z;
        tmin = fmax_wrapper(tmin, fmin_wrapper(tz1, tz2));
        tmax = fmin_wrapper(tmax, fmax_wrapper(tz1, tz2));

        return (tmax >= tmin && tmax > t_min && tmin < t_max);
    }
};
