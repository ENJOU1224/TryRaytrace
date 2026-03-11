#pragma once 
#include "common.h" // 需要 Vec 定义
#include "aabb.h"
#include <string>
#include <vector>

// ======================================================================================
// 1. 基础枚举定义
// ======================================================================================

// [材质类型]
// 影响光线反弹的行为 (BSDF)
enum Refl_t { 
    DIFF, // 漫反射 (Diffuse): 朗伯余弦分布，模拟粗糙表面
    SPEC, // 镜面反射 (Specular): 完美反射，模拟镜子/金属
    REFR  // 折射 (Refractive): 斯涅尔定律 + 菲涅尔效应，模拟玻璃/水
};

// ======================================================================================
// 2. 物体结构体 (Object) - 几何预计算版
// ======================================================================================
// [优化策略]
// 1. 继续保持 16 字节对齐，让 GPU 读取 Vec 时更稳定。
// 2. 将三角形的静态几何量 (edge1 / edge2 / normal / area) 预烘焙到对象里，
//    避免每次光线求交、阴影测试、灯采样都重复计算。
// 3. 这会增加一些显存占用，但对当前路径追踪器来说，减少 kernel 内重复算术更划算。
// --------------------------------------------------------------------------------------
struct ALIGN(16) Object {
    Vec v0 = {0.0f, 0.0f, 0.0f};
    Vec edge1 = {0.0f, 0.0f, 0.0f};
    Vec edge2 = {0.0f, 0.0f, 0.0f};
    Vec normal = {0.0f, 0.0f, 0.0f};
    Vec albedo = {0.0f, 0.0f, 0.0f};
    Vec emission = {0.0f, 0.0f, 0.0f};

    float metallic = 0.0f;
    float roughness = 1.0f;
    float ior = 1.45f;
    float transmission = 0.0f;
    float area = 0.0f;

    int tex_id = -1;
    
    float pad1 = 0.0f;
    float pad2 = 0.0f;
};

/**
 * @brief 预计算三角形静态几何量
 *
 * 这些量只依赖顶点位置，不依赖相机或随机数。
 * 因此把它们前移到 CPU 端烘焙，能减少 GPU kernel 中的大量重复计算。
 */
inline void bake_object_geometry(Object& obj) {
    Vec geometric_normal = obj.edge1.cross(obj.edge2);
    const float double_area = geometric_normal.norm_len();
    obj.area = 0.5f * double_area;

    if (double_area > 1e-12f) {
        obj.normal = geometric_normal * (1.0f / double_area);
    } else {
        obj.normal = {0.0f, 0.0f, 0.0f};
    }
}

/**
 * @brief 创建一个带预计算几何量的三角形对象
 */
inline Object make_object(Vec v0,
                          Vec v1,
                          Vec v2,
                          Vec albedo,
                          Vec emission,
                          float metallic,
                          float roughness,
                          float ior = 1.45f,
                          float transmission = 0.0f,
                          int tex_id = -1) {
    Object obj;
    obj.v0 = v0;
    obj.edge1 = v1 - v0;
    obj.edge2 = v2 - v0;
    obj.albedo = albedo;
    obj.emission = emission;
    obj.metallic = metallic;
    obj.roughness = roughness;
    obj.ior = ior;
    obj.transmission = transmission;
    obj.tex_id = tex_id;
    bake_object_geometry(obj);
    return obj;
}

// ======================================================================================
// 3. 全局配置与相机
// ======================================================================================

// [相机参数]
// 这个结构体在每一帧开始时由 CPU 传给 GPU Kernel。
// 包含生成光线所需的所有几何信息。
struct CameraParams {
    Vec pos; // 相机世界坐标
    Vec cx;  // 成像平面 X 轴 (已包含 FOV 缩放)
    Vec cy;  // 成像平面 Y 轴 (已包含 FOV 缩放)
    Vec dir; // 相机朝向 (归一化)
    
    float lens_radius; // 光圈半径 (Aperture / 2)。0 为针孔相机。
    float focus_dist;  // 焦距。光线在何处汇聚。
};

// ======================================================================================
// 4. 场景管理接口
// ======================================================================================

// 这是一个纯 CPU 端的数据容器，负责管理资源的生命周期。
// 使用 std::vector 可以方便地动态添加物体。
struct Scene {
    std::vector<Object> objects;
    std::vector<std::string> texture_files;

    AABB world_bound;
};

// 工厂函数
Scene create_cornell_box();
