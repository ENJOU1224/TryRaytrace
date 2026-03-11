#pragma once

#include <cmath>
#include <algorithm>

// 处理 CUDA 和 SYCL 的兼容性宏
#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #define HOST_DEVICE __host__ __device__
    #define ALIGN(n) __align__(n)
#elif defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
    #include <sycl/sycl.hpp>
    #define HOST_DEVICE
    #define ALIGN(n) alignas(n)
#else
    #define HOST_DEVICE
    #define ALIGN(n) alignas(n)
    #ifndef __align__
        #define __align__(n) alignas(n)
    #endif
#endif

// [兼容性宏]
// M_PI 是圆周率。虽然标准库通常有，但在某些编译器设置下可能未定义。
// 为了保证全平台(Windows/Linux)通用，我们手动定义它。
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ======================================================================================
// 核心数据结构: Vec (向量/颜色)
// ======================================================================================
// [硬件优化关键点]: __align__(16)
// 1. 性能: GPU 的显存控制器喜欢读取 128位(16字节) 的数据块。
//    普通的 float x,y,z 只有 12字节。强制对齐到 16字节会让编译器填充 4字节 空缺。
//    这样 GPU 读取一个 Vec 只需要 1 个指令周期，而不是拆成多次。
// 2. 正确性: 强制 CPU 和 GPU 使用相同的内存布局。
//    防止因为编译器默认对齐策略不同，导致 CPU 传给 GPU 的数据错位 (比如我们之前遇到的镜面球变玻璃球的问题)。
struct ALIGN(16) Vec {
    float x, y, z; 

    HOST_DEVICE Vec operator+(const Vec& b) const { 
        return {x + b.x, y + b.y, z + b.z}; 
    }

    HOST_DEVICE Vec operator-(const Vec& b) const { 
        return {x - b.x, y - b.y, z - b.z}; 
    }

    HOST_DEVICE Vec operator*(float b) const { 
        return {x * b, y * b, z * b}; 
    }

    HOST_DEVICE Vec mult(const Vec& b) const { 
        return {x * b.x, y * b.y, z * b.z}; 
    }

    HOST_DEVICE Vec& norm() { 
#if defined(__SYCL_DEVICE_ONLY__)
        float len = sycl::native::sqrt(x * x + y * y + z * z);
#else
        float len = std::sqrt(x * x + y * y + z * z);
#endif
        if (len > 0) { 
            float invLen = 1.0f / len;
            x *= invLen; y *= invLen; z *= invLen; 
        }
        return *this;
    }

    HOST_DEVICE float dot(const Vec& b) const { 
#if defined(__SYCL_DEVICE_ONLY__)
        return sycl::fma(x, b.x, sycl::fma(y, b.y, z * b.z));
#else
        return x * b.x + y * b.y + z * b.z; 
#endif
    }

    HOST_DEVICE Vec cross(const Vec& b) const { 
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x}; 
    }

    HOST_DEVICE float norm_len() const { return std::sqrt(x*x + y*y + z*z); }
};

HOST_DEVICE inline Vec make_vec(float x, float y, float z) { 
    Vec v = {x, y, z}; 
    return v; 
}

// [辅助函数] 钳制函数 (Clamp)
// 作用: 防止颜色溢出。
// 光线追踪计算出的亮度可能超过 1.0 (比如直视太阳)，也可能因为误差略小于 0.0。
// 在存图之前，必须把它限制在 [0, 1] 范围内。
inline float clamp(float x) { 
    return x < 0 ? 0 : x > 1 ? 1 : x; 
}

// [辅助函数] Reinhard Tone Mapping
// 作用: 将 HDR 线性亮度压缩到显示器更容易承载的范围。
// 为什么需要它？
// 1. 路径追踪天然会产生大于 1.0 的高动态范围亮度。
// 2. 如果直接 clamp，再做 Gamma，所有超过 1.0 的值都会被硬裁成纯白，
//    肉眼看上去就像全屏随机冒出的白点。
// 3. Reinhard 是最简单稳定的基线：x / (1 + x)。
inline float tone_map_reinhard(float x) {
    float safe = x < 0.0f ? 0.0f : x;
    return safe / (1.0f + safe);
}

// [辅助函数] 显示编码
// 作用: 将 HDR 线性亮度转换为显示器使用的 SDR 值。
// 核心步骤:
// 1. tone_map_reinhard(x): 压缩动态范围。
// 2. clamp(x): 保证安全范围。
// 3. pow(..., 1/2.2): Gamma 校正。
//    显示器是非线性的，会把画面压暗。我们需要预先将画面"提亮" (指数 0.45)，
//    这样显示器压暗后，人眼看到的才是正确的物理亮度。
inline float encode_display_value(float x) {
    return std::pow(clamp(tone_map_reinhard(x)), 1.0f / 2.2f);
}

// [辅助函数] 颜色量化
// 作用: 将经过显示编码的浮点亮度转换为 0 - 255 整数。
inline int toInt(float x) { 
    return int(encode_display_value(x) * 255 + .5f); 
}
