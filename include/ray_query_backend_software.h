#pragma once

namespace ray_query_backend {

// 当前默认后端：软件 BVH 遍历 + 软件三角形求交。
// 后续如果要切 Embree GPU，可以保留同样的函数签名，
// 并把 ray_query_backend.h 末尾 include 的实现替换掉。

inline void initialize_scene_backend(sycl::queue&,
                                     const std::vector<Object>&,
                                     const std::vector<LinearBVHNode>&) {}

inline void shutdown_scene_backend(sycl::queue&) {}

inline void set_scene_view_backend_state(SceneView& scene) {
    scene.backend_handle = 0;
    scene.primitive_count = 0;
}

HOST_DEVICE inline bool intersect_triangle(const Vec& ray_origin,
                                           const Vec& ray_dir,
                                           const Object& obj,
                                           float t_min,
                                           float t_max,
                                           float& out_t) {
    Vec h = ray_dir.cross(obj.edge2);
    float a = obj.edge1.dot(h);
    if (a > -1e-5f && a < 1e-5f) {
        return false;
    }

    float f = 1.0f / a;
    Vec s = ray_origin - obj.v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    Vec q = s.cross(obj.edge1);
    float v = f * ray_dir.dot(q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = f * obj.edge2.dot(q);
    if (t <= t_min || t >= t_max) {
        return false;
    }

    out_t = t;
    return true;
}

HOST_DEVICE inline SceneHit find_closest_hit(const SceneView& scene,
                                             const Vec& ray_origin,
                                             const Vec& ray_dir,
                                             float t_min) {
    SceneHit hit;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    Vec ray_inv_dir = {1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z};

    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        const auto& node = scene.bvh_nodes[idx];
        if (!node.bounds.hit(ray_origin, ray_inv_dir, t_min, hit.t)) {
            continue;
        }

        if (node.is_leaf) {
            for (int k = 0; k < node.primitive_count; ++k) {
                int obj_idx = node.primitive_offset + k;
                float t = hit.t;
                if (intersect_triangle(ray_origin,
                                       ray_dir,
                                       scene.objects[obj_idx],
                                       t_min,
                                       hit.t,
                                       t)) {
                    hit.t = t;
                    hit.object_id = obj_idx;
                }
            }
        } else {
            stack[stack_ptr++] = node.right_child_idx;
            stack[stack_ptr++] = node.left_child_idx;
        }
    }

    return hit;
}

HOST_DEVICE inline bool is_shadowed(const SceneView& scene,
                                    const Vec& shadow_origin,
                                    const Vec& shadow_dir,
                                    float t_min,
                                    float max_distance) {
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    Vec inv_dir = {1.0f / shadow_dir.x, 1.0f / shadow_dir.y, 1.0f / shadow_dir.z};

    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        if (!scene.bvh_nodes[idx].bounds.hit(shadow_origin, inv_dir, t_min, max_distance - 0.01f)) {
            continue;
        }

        if (scene.bvh_nodes[idx].is_leaf) {
            for (int k = 0; k < scene.bvh_nodes[idx].primitive_count; ++k) {
                int object_id = scene.bvh_nodes[idx].primitive_offset + k;
                float t = max_distance;
                if (intersect_triangle(shadow_origin,
                                       shadow_dir,
                                       scene.objects[object_id],
                                       t_min,
                                       max_distance - 0.01f,
                                       t)) {
                    return true;
                }
            }
        } else {
            stack[stack_ptr++] = scene.bvh_nodes[idx].right_child_idx;
            stack[stack_ptr++] = scene.bvh_nodes[idx].left_child_idx;
        }
    }

    return false;
}

}  // namespace ray_query_backend
