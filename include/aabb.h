#pragma once
#include "common.h"

// 兼容 CPU/GPU 的 min/max 函数
HOST_DEVICE inline float fmin_wrapper(float a, float b) { return a < b ? a : b; }
HOST_DEVICE inline float fmax_wrapper(float a, float b) { return a > b ? a : b; }

// ======================================================================================
// AABB (轴对齐包围盒)
// ======================================================================================
struct ALIGN(16) AABB {
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
