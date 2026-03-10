#include "renderer.h"
#include "common.h"
#include "aabb.h"
#include "bvh.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

static queue* g_queue = nullptr;
static Object* d_objects = nullptr;
static LinearBVHNode* d_bvh_nodes = nullptr;
static int* d_light_indices = nullptr;
static int d_light_count = 0;

HOST_DEVICE unsigned int hash(unsigned int x) {
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

struct Random {
    unsigned int state;
    HOST_DEVICE Random(unsigned int pixel_idx, unsigned int seed) {
        state = hash(pixel_idx) ^ hash(seed);
    }
    HOST_DEVICE unsigned int next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    HOST_DEVICE float next_float() {
        return (float)next() * 2.3283064365386963e-10f; 
    }
};

HOST_DEVICE Vec fresnel_schlick(float cos_theta, Vec f0) {
    float x = 1.0f - cos_theta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return f0 + (Vec{1,1,1} - f0) * x5;
}

HOST_DEVICE float intersect_triangle(const Object& obj, const Vec& r_o, const Vec& r_d) {
    const float eps = 1e-5f;
    Vec e1 = obj.v1 - obj.v0, e2 = obj.v2 - obj.v0;
    Vec h = r_d.cross(e2); 
    float a = e1.dot(h);
    if (a > -eps && a < eps) return 0.0f;
    float f = 1.0f / a; 
    Vec s = r_o - obj.v0;
    float u = f * s.dot(h); 
    if (u < 0.0f || u > 1.0f) return 0.0f;
    Vec q = s.cross(e1); 
    float v = f * r_d.dot(q);
    if (v < 0.0f || u + v > 1.0f) return 0.0f;
    float t = f * e2.dot(q); 
    return (t > eps) ? t : 0.0f;
}

HOST_DEVICE Vec trace(Vec r_o, Vec r_d, Random& rng, const LinearBVHNode* bvh_nodes, const Object* scene_objects, const int* light_indices, int light_count) {
    Vec radiance = {0, 0, 0}, throughput = {1, 1, 1};
    const int MAX_DEPTH = 10;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        float d_min = 1e20f; int id = -1;
        int stack[32]; int ptr = 0; stack[ptr++] = 0;
        Vec r_inv_d = {1.0f/r_d.x, 1.0f/r_d.y, 1.0f/r_d.z};

        while (ptr > 0) {
            int idx = stack[--ptr]; const auto& node = bvh_nodes[idx];
            if (!node.bounds.hit(r_o, r_inv_d, 0.001f, d_min)) continue;
            if (node.is_leaf) {
                for (int k = 0; k < node.primitive_count; k++) {
                    int obj_idx = node.primitive_offset + k;
                    float t = intersect_triangle(scene_objects[obj_idx], r_o, r_d);
                    if (t > 0 && t < d_min) { d_min = t; id = obj_idx; }
                }
            } else { 
                stack[ptr++] = node.right_child_idx; stack[ptr++] = node.left_child_idx; 
            }
        }

        if (id < 0) break;

        const Object& obj = scene_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        Vec n = (obj.v1 - obj.v0).cross(obj.v2 - obj.v0).norm();
        Vec nl = n.dot(r_d) < 0 ? n : n * -1;

        radiance = radiance + throughput.mult(obj.emission);
        if (obj.emission.norm_len() > 0.1f) break;

        Vec albedo = obj.albedo;
        float metallic = obj.metallic, roughness = obj.roughness, transmission = obj.transmission;
        
        float dot_val = r_d.dot(nl);
        float cos_theta = (dot_val < 0) ? -dot_val : dot_val;

        Vec f0 = albedo * metallic + Vec{0.04f, 0.04f, 0.04f} * (1.0f - metallic);
        Vec F = fresnel_schlick(cos_theta, f0);
        float p_spec = (F.x + F.y + F.z) * 0.3333f;

        float rnd = rng.next_float();
        if (rnd < transmission) {
            float ior = obj.ior > 0 ? obj.ior : 1.5f;
            bool into = n.dot(nl) > 0; float nnt = into ? 1.0f / ior : ior;
            float ddn = r_d.dot(nl); 
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
            if (cos2t < 0) {
                r_d = (r_d - n * 2 * r_d.dot(n)).norm();
            } else {
                r_d = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + std::sqrt(cos2t)))).norm();
            }
            r_o = x_hit + r_d * 0.001f; 
            float p_branch = (transmission > 0.01f) ? transmission : 0.01f;
            throughput = throughput.mult(albedo) * (1.0f / p_branch);
        } else if (rnd < transmission + p_spec) {
            Vec perfect = (r_d - n * 2 * r_d.dot(n)).norm();
            float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float();
            float z = 1.0f - 2.0f * r2;
            float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
            Vec rand_v = {r * std::cos(r1), r * std::sin(r1), z};
            r_d = (perfect + rand_v * roughness).norm();
            if(r_d.dot(nl) < 0) break;
            r_o = x_hit + nl * 0.001f; 
            float p_branch = (p_spec > 0.01f) ? p_spec : 0.01f;
            throughput = throughput.mult(F) * (1.0f / p_branch);
        } else {
            if (light_count > 0 && depth < 3) {
                const Object& light = scene_objects[light_indices[(int)(rng.next_float() * light_count)]];
                float r_u = rng.next_float();
                float lu = 1.0f - std::sqrt(r_u);
                float lv = rng.next_float() * (1.0f - lu);
                Vec lp = light.v0 * lu + light.v1 * lv + light.v2 * (1.0f - lu - lv);
                Vec tl = lp - x_hit; 
                float dist_sq = tl.dot(tl);
                float dist = std::sqrt(dist_sq);
                Vec ld = tl * (1.0f / dist);
                if (nl.dot(ld) > 0) {
                    bool blocked = false; int s_ptr = 0, s_stack[32]; s_stack[s_ptr++] = 0; 
                    Vec s_inv = {1.0f/ld.x, 1.0f/ld.y, 1.0f/ld.z};
                    while(s_ptr > 0) {
                        int idx = s_stack[--s_ptr]; const auto& node = bvh_nodes[idx];
                        if(!node.bounds.hit(x_hit + nl * 0.001f, s_inv, 0.001f, dist - 0.01f)) continue;
                        if(node.is_leaf) {
                            for(int k=0; k<node.primitive_count; k++)
                                if(intersect_triangle(scene_objects[node.primitive_offset+k], x_hit + nl * 0.001f, ld) > 0) { blocked = true; break; }
                            if(blocked) break;
                        } else { s_stack[s_ptr++] = node.right_child_idx; s_stack[s_ptr++] = node.left_child_idx; }
                    }
                    if (!blocked) {
                        float area = (light.v1 - light.v0).cross(light.v2 - light.v0).norm_len() * 0.5f;
                        Vec ln = (light.v1 - light.v0).cross(light.v2 - light.v0).norm();
                        float dot_ln = ln.dot(ld * -1.0f);
                        float cos_l = (dot_ln < 0) ? -dot_ln : dot_ln;
                        radiance = radiance + throughput.mult(light.emission.mult(albedo)) * (nl.dot(ld) * cos_l * area / (dist_sq * M_PI * (1.0f/light_count)));
                    }
                }
            }
            float r1 = 2 * M_PI * rng.next_float(), r2 = rng.next_float();
            float r2s = std::sqrt(r2);
            Vec w = nl, basis_u = (( (w.x > 0 ? w.x : -w.x) > 0.1f ? Vec{0,1,0} : Vec{1,0,0}).cross(w)).norm(), basis_v = w.cross(basis_u);
            r_d = (basis_u * std::cos(r1) * r2s + basis_v * std::sin(r1) * r2s + w * std::sqrt(std::max(0.0f, 1.0f-r2))).norm();
            r_o = x_hit + nl * 0.001f; 
            float p_diff = 1.0f - transmission - p_spec;
            p_diff = (p_diff > 0.01f) ? p_diff : 0.01f;
            throughput = throughput.mult(albedo) * (1.0f / p_diff);
        }
        if (depth > 3) {
            float p = albedo.x > albedo.y ? (albedo.x > albedo.z ? albedo.x : albedo.z) : (albedo.y > albedo.z ? albedo.y : albedo.z);
            if (p < 0.1f) p = 0.1f;
            if (rng.next_float() > p) break;
            throughput = throughput * (1.0f / p);
        }
    }
    return radiance;
}

void init_renderer_sycl() {
    if (!g_queue) {
        try { g_queue = new queue(gpu_selector_v); } 
        catch (...) { g_queue = new queue(cpu_selector_v); }
    }
}

void init_scene_data(const std::vector<Object>& objects, const std::vector<std::string>& texture_files, const std::vector<LinearBVHNode>& nodes, const std::vector<int>& light_indices) {
    init_renderer_sycl();
    if (d_objects) free(d_objects, *g_queue);
    d_objects = malloc_shared<Object>(objects.size(), *g_queue);
    std::memcpy(d_objects, objects.data(), objects.size() * sizeof(Object));
    if (d_bvh_nodes) free(d_bvh_nodes, *g_queue);
    d_bvh_nodes = malloc_shared<LinearBVHNode>(nodes.size(), *g_queue);
    std::memcpy(d_bvh_nodes, nodes.data(), nodes.size() * sizeof(LinearBVHNode));
    d_light_count = light_indices.size();
    if (d_light_indices) free(d_light_indices, *g_queue);
    if (d_light_count > 0) {
        d_light_indices = malloc_shared<int>(d_light_count, *g_queue);
        std::memcpy(d_light_indices, light_indices.data(), d_light_count * sizeof(int));
    }
}

void launch_render_kernel(Vec* accum_buffer_usm, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    auto objects = d_objects; auto nodes = d_bvh_nodes; auto lights = d_light_indices; int l_count = d_light_count;
    g_queue->submit([&](handler& h) {
        h.parallel_for(nd_range<2>(range<2>(width, height), range<2>(tx, ty)), [=](nd_item<2> item) {
            int x = item.get_global_id(0), y = item.get_global_id(1);
            if (x >= width || y >= height) return;
            int i = y * width + x; 
            Random rng(i, frame_seed);
            float fx = (float)(x + rng.next_float() - 0.5f) / width - 0.5f;
            float fy = 0.5f - (float)(y + rng.next_float() - 0.5f) / height;
            Vec r_d = (cam.cx * fx + cam.cy * fy + cam.dir).norm();
            Vec color = trace(cam.pos, r_d, rng, nodes, objects, lights, l_count);
            if (std::isnan(color.x) || std::isinf(color.x)) color = {0,0,0};
            float lum = color.x * 0.21f + color.y * 0.71f + color.z * 0.07f;
            if (lum > 10.0f) color = color * (10.0f / lum); 
            accum_buffer_usm[i] = accum_buffer_usm[i] + color;
        });
    });
    g_queue->wait();
}
