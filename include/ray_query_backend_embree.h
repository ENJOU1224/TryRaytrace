#pragma once

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

// 这个文件实现 Embree GPU 查询后端。
//
// 设计目标不是一口气把整个渲染器“Embree 化”，而是只替换：
// - 最近交点查询
// - 阴影遮挡查询
//
// 路径追踪主循环、材质逻辑、光源采样和 firefly 处理仍然保留在 renderer_sycl.cpp。

namespace ray_query_backend {
namespace {

inline RTCDevice g_embree_device = nullptr;
inline RTCScene g_embree_scene = nullptr;
inline RTCTraversable g_embree_traversable = nullptr;
// Embree 场景本身不直接复用我们的 Object 内存布局，
// 所以这里另外准备一份适合 Embree 三角形接口的顶点/索引缓冲。
inline float* g_embree_vertices = nullptr;
inline uint32_t* g_embree_indices = nullptr;
inline int g_embree_primitive_count = 0;

inline constexpr RTCFeatureFlags kEmbreeRequiredFeatures = RTC_FEATURE_FLAG_TRIANGLE;
inline constexpr float kShadowDistanceEpsilon = 0.01f;

inline void embree_error_function(void*, RTCError error, const char* message) {
    std::cerr << "[Embree] " << rtcGetErrorString(error) << ": "
              << (message ? message : "(no message)") << std::endl;
}

inline std::runtime_error make_embree_error(const char* prefix) {
    const RTCError error = rtcGetDeviceError(nullptr);
    const char* message = rtcGetDeviceLastErrorMessage(nullptr);
    std::string text = prefix;
    text += ": ";
    text += rtcGetErrorString(error);
    if (message && message[0] != '\0') {
        text += " | ";
        text += message;
    }
    return std::runtime_error(text);
}

inline RTCTraversable embree_traversable_from_handle(std::uintptr_t handle) {
    return reinterpret_cast<RTCTraversable>(handle);
}

}  // namespace

inline void shutdown_scene_backend(sycl::queue& queue) {
    // 注意释放顺序：
    // 1. 先释放 Embree 场景/设备句柄
    // 2. 再释放我们手动申请的 USM 顶点/索引缓冲
    // 这样最安全，也更符合“谁先持有逻辑资源，谁先被释放”的思路。
    g_embree_traversable = nullptr;
    g_embree_primitive_count = 0;

    if (g_embree_scene) {
        rtcReleaseScene(g_embree_scene);
        g_embree_scene = nullptr;
    }
    if (g_embree_device) {
        rtcReleaseDevice(g_embree_device);
        g_embree_device = nullptr;
    }
    if (g_embree_vertices) {
        sycl::free(g_embree_vertices, queue);
        g_embree_vertices = nullptr;
    }
    if (g_embree_indices) {
        sycl::free(g_embree_indices, queue);
        g_embree_indices = nullptr;
    }
}

inline void initialize_scene_backend(sycl::queue& queue,
                                     const std::vector<Object>& objects,
                                     const std::vector<LinearBVHNode>&) {
    shutdown_scene_backend(queue);

    if (objects.empty()) {
        return;
    }

    const sycl::device device = queue.get_device();
    if (!rtcIsSYCLDeviceSupported(device)) {
        throw std::runtime_error("Embree does not support the renderer SYCL device.");
    }

    // rtcNewSYCLDevice 会基于当前 SYCL context 创建 Embree 设备。
    // 后面所有几何、场景、遍历句柄都从这个 device 派生。
    g_embree_device = rtcNewSYCLDevice(queue.get_context(), "");
    if (!g_embree_device) {
        throw make_embree_error("rtcNewSYCLDevice failed");
    }

    rtcSetDeviceSYCLDevice(g_embree_device, device);
    rtcSetDeviceErrorFunction(g_embree_device, embree_error_function, nullptr);

    g_embree_scene = rtcNewScene(g_embree_device);
    RTCGeometry geometry = rtcNewGeometry(g_embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    const size_t triangle_count = objects.size();
    const size_t vertex_count = triangle_count * 3;
    g_embree_vertices = sycl::malloc_shared<float>(vertex_count * 3, queue);
    g_embree_indices = sycl::malloc_shared<uint32_t>(triangle_count * 3, queue);
    if (!g_embree_vertices || !g_embree_indices) {
        rtcReleaseGeometry(geometry);
        shutdown_scene_backend(queue);
        throw std::bad_alloc();
    }

    for (size_t i = 0; i < triangle_count; ++i) {
        const Object& obj = objects[i];
        // 我们自己的 Object 存的是：
        //   v0 + edge1 + edge2
        // Embree 需要的是：
        //   明确展开后的 v0 / v1 / v2 顶点数组
        const Vec v0 = obj.v0;
        const Vec v1 = obj.v0 + obj.edge1;
        const Vec v2 = obj.v0 + obj.edge2;

        const size_t vertex_base = i * 9;
        g_embree_vertices[vertex_base + 0] = v0.x;
        g_embree_vertices[vertex_base + 1] = v0.y;
        g_embree_vertices[vertex_base + 2] = v0.z;
        g_embree_vertices[vertex_base + 3] = v1.x;
        g_embree_vertices[vertex_base + 4] = v1.y;
        g_embree_vertices[vertex_base + 5] = v1.z;
        g_embree_vertices[vertex_base + 6] = v2.x;
        g_embree_vertices[vertex_base + 7] = v2.y;
        g_embree_vertices[vertex_base + 8] = v2.z;

        const size_t index_base = i * 3;
        g_embree_indices[index_base + 0] = static_cast<uint32_t>(i * 3 + 0);
        g_embree_indices[index_base + 1] = static_cast<uint32_t>(i * 3 + 1);
        g_embree_indices[index_base + 2] = static_cast<uint32_t>(i * 3 + 2);
    }

    rtcSetSharedGeometryBuffer(geometry,
                               RTC_BUFFER_TYPE_VERTEX,
                               0,
                               RTC_FORMAT_FLOAT3,
                               g_embree_vertices,
                               0,
                               3 * sizeof(float),
                               vertex_count);
    rtcSetSharedGeometryBuffer(geometry,
                               RTC_BUFFER_TYPE_INDEX,
                               0,
                               RTC_FORMAT_UINT3,
                               g_embree_indices,
                               0,
                               3 * sizeof(uint32_t),
                               triangle_count);

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(g_embree_scene, geometry);
    rtcReleaseGeometry(geometry);

    // rtcCommitScene 会真正构建 Embree 使用的加速结构。
    // commit 完之后，再取新的 RTCTraversable 句柄给 GPU kernel 使用。
    rtcCommitScene(g_embree_scene);
    g_embree_traversable = rtcGetSceneTraversable(g_embree_scene);
    g_embree_primitive_count = static_cast<int>(triangle_count);
}

inline void set_scene_view_backend_state(SceneView& scene) {
    scene.backend_handle = reinterpret_cast<std::uintptr_t>(g_embree_traversable);
    scene.primitive_count = g_embree_primitive_count;
}

HOST_DEVICE inline SceneHit find_closest_hit(const SceneView& scene,
                                             const Vec& ray_origin,
                                             const Vec& ray_dir,
                                             float t_min) {
    SceneHit hit;
    RTCTraversable traversable = embree_traversable_from_handle(scene.backend_handle);
    if (!traversable) {
        return hit;
    }

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.feature_mask = kEmbreeRequiredFeatures;

    RTCRayHit rayhit;
    // 这组字段和我们原先软件求交里的“射线参数”一一对应。
    rayhit.ray.org_x = ray_origin.x;
    rayhit.ray.org_y = ray_origin.y;
    rayhit.ray.org_z = ray_origin.z;
    rayhit.ray.dir_x = ray_dir.x;
    rayhit.ray.dir_y = ray_dir.y;
    rayhit.ray.dir_z = ray_dir.z;
    rayhit.ray.tnear = t_min;
    rayhit.ray.tfar = 1e30f;
    rayhit.ray.mask = 0xFFFFFFFFu;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcTraversableIntersect1(traversable, &rayhit, &args);

    // 当前实现里，一个 Embree primitive 对应一个 Object 三角形，
    // 所以 primID 可以直接映射回对象下标。
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID &&
        rayhit.hit.primID != RTC_INVALID_GEOMETRY_ID &&
        static_cast<int>(rayhit.hit.primID) < scene.primitive_count) {
        hit.t = rayhit.ray.tfar;
        hit.object_id = static_cast<int>(rayhit.hit.primID);
    }
    return hit;
}

HOST_DEVICE inline bool is_shadowed(const SceneView& scene,
                                    const Vec& shadow_origin,
                                    const Vec& shadow_dir,
                                    float t_min,
                                    float max_distance) {
    RTCTraversable traversable = embree_traversable_from_handle(scene.backend_handle);
    if (!traversable) {
        return false;
    }

    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.feature_mask = kEmbreeRequiredFeatures;

    RTCRay ray;
    ray.org_x = shadow_origin.x;
    ray.org_y = shadow_origin.y;
    ray.org_z = shadow_origin.z;
    ray.dir_x = shadow_dir.x;
    ray.dir_y = shadow_dir.y;
    ray.dir_z = shadow_dir.z;
    ray.tnear = t_min;
    // 这里必须像软件后端一样，把终点往回收一点。
    // 否则“正好打到采样到的灯面”也会被 Embree 记成遮挡，
    // 结果就是直接光大量消失，整幅图明显发暗。
    ray.tfar = max_distance - kShadowDistanceEpsilon;
    if (ray.tfar <= ray.tnear) {
        return false;
    }
    ray.mask = 0xFFFFFFFFu;
    ray.flags = 0;

    rtcTraversableOccluded1(traversable, &ray, &args);
    // Embree 的 occluded 查询会把 tfar 设成负值来表示“被遮挡”。
    return ray.tfar < 0.0f;
}

}  // namespace ray_query_backend
