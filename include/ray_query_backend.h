#pragma once

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "common.h"
#include "scene.h"
#include "bvh.h"

// 这个头文件定义“渲染器真正依赖的几何查询接口”。
//
// 当前实现仍然是软件 BVH 遍历，但我们把接口边界先固定下来：
// - SceneView: 渲染阶段可见的场景查询视图
// - SceneHit:   最近命中结果
// - find_closest_hit(): 最近交点查询
// - is_shadowed():      阴影遮挡查询
//
// 后续如果要接 Embree GPU，优先替换的应该就是这里 include 的后端实现，
// 而不是把 renderer_sycl.cpp 里的材质和路径逻辑整体推翻。
namespace ray_query_backend {

#if defined(RAY_QUERY_BACKEND_SOFTWARE)
inline constexpr const char* kSelectedBackendName = "software";
#elif defined(RAY_QUERY_BACKEND_EMBREE)
inline constexpr const char* kSelectedBackendName = "embree";
#else
inline constexpr const char* kSelectedBackendName = "unknown";
#endif

struct SceneHit {
    // 最近交点到射线原点的距离。
    float t = 1e20f;
    // 命中的三角形对象编号。-1 表示没有命中。
    int object_id = -1;
};

struct SceneView {
    // 当前材质、光照和显示链仍然直接使用 Object 数组。
    const Object* objects = nullptr;

    // 当前软件后端使用线性 BVH。
    // 如果未来换成别的查询后端，可以把这里当作“当前后端自己的几何查询句柄入口”。
    const LinearBVHNode* bvh_nodes = nullptr;

    // 光源索引列表依然保留在 SceneView 中，
    // 因为路径追踪的 NEE 逻辑仍然由 renderer_sycl.cpp 自己控制。
    const int* light_indices = nullptr;
    int light_count = 0;

    // backend_handle 是当前查询后端自己的句柄。
    // - software 后端里它为空
    // - embree 后端里它会保存 RTCTraversable
    //
    // 这里故意不用具体类型，是为了让 renderer_sycl.cpp 不需要直接 include Embree 头文件，
    // 从而把“主渲染逻辑”和“具体几何查询后端”继续隔离开。
    std::uintptr_t backend_handle = 0;

    // 记录当前后端里一共对应了多少个可查询 primitive。
    // Embree 后端用它把 primID 映射回 Object 下标。
    int primitive_count = 0;
};

// 这三个函数负责“后端自己的资源生命周期”：
// 1. initialize_scene_backend: 用当前后端格式准备场景查询数据
// 2. shutdown_scene_backend:   释放后端资源
// 3. set_scene_view_backend_state: 把后端句柄填进 SceneView，供 kernel 查询
void initialize_scene_backend(sycl::queue& queue,
                              const std::vector<Object>& objects,
                              const std::vector<LinearBVHNode>& nodes);

void shutdown_scene_backend(sycl::queue& queue);

void set_scene_view_backend_state(SceneView& scene);

HOST_DEVICE SceneHit find_closest_hit(const SceneView& scene,
                                      const Vec& ray_origin,
                                      const Vec& ray_dir,
                                      float t_min);

HOST_DEVICE bool is_shadowed(const SceneView& scene,
                             const Vec& shadow_origin,
                             const Vec& shadow_dir,
                             float t_min,
                             float max_distance);

}  // namespace ray_query_backend

// ------------------------------------------------------------------
// 后端选择
// ------------------------------------------------------------------
// 默认使用软件后端。
// 这样当前项目不会因为 Embree 未安装而影响现有流程。
#if !defined(RAY_QUERY_BACKEND_SOFTWARE) && !defined(RAY_QUERY_BACKEND_EMBREE)
#define RAY_QUERY_BACKEND_SOFTWARE 1
#endif

#if defined(RAY_QUERY_BACKEND_SOFTWARE) && defined(RAY_QUERY_BACKEND_EMBREE)
#error "Only one ray query backend can be enabled at a time."
#endif

#if defined(RAY_QUERY_BACKEND_SOFTWARE)
#include "ray_query_backend_software.h"
#elif defined(RAY_QUERY_BACKEND_EMBREE)
#include "ray_query_backend_embree.h"
#else
#error "No ray query backend selected."
#endif
