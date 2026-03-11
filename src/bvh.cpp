#include "bvh.h"
#include <algorithm>
#include <iostream>

namespace {

// 对 GPU 路径追踪来说，“每个叶子只放 1 个三角形”并不一定最优。
// 叶子过细会让树更深、内部节点更多、栈操作更多。
// 当前先用一个保守值 4 做实验，通常能在“少走树”和“叶子里多测几个三角形”之间取得更平衡的结果。
constexpr int kLeafPrimitiveCount = 4;
constexpr int kSahBinCount = 12;

float axis_value(const Vec& v, int axis) {
    if (axis == 0) return v.x;
    if (axis == 1) return v.y;
    return v.z;
}

float surface_area(const AABB& box) {
    Vec size = box.max - box.min;
    return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

struct SahBin {
    AABB bounds = AABB::empty();
    int count = 0;
};

}

// 获取物体的中心点 (用于排序)
Vec get_centroid(const Object& obj) {
    // 三角形重心
    return (obj.v0 + (obj.v0 + obj.edge1) + (obj.v0 + obj.edge2)) * 0.333333f;
}

// 获取物体的 AABB
AABB get_object_bounds(const Object& obj) {
    AABB box = AABB::empty();

    box.grow(obj.v0);
    box.grow(obj.v0 + obj.edge1);
    box.grow(obj.v0 + obj.edge2);

    // [关键修复] 防止扁平三角形导致 AABB 厚度为 0 (数学黑洞)
    // 给每个轴加一个微小的厚度 (Padding)
    const float pad = 1e-3f;
    Vec size = box.max - box.min;
    
    // 如果某个轴压扁了，强行撑开一点
    if (size.x < pad) { box.min.x -= pad; box.max.x += pad; }
    if (size.y < pad) { box.min.y -= pad; box.max.y += pad; } 
    if (size.z < pad) { box.min.z -= pad; box.max.z += pad; }

    return box;
}

void BVH::build(std::vector<Object>& objects) {
    nodes.clear();
    // 预留空间，防止频繁 realloc
    // 二叉树节点数通常是物体数的 2 倍左右
    nodes.reserve(objects.size() * 2);

    if (objects.empty()) return;

    // 开始递归构建
    printf("[BVH] Building BVH for %lu objects...\n", objects.size());
    build_recursive(objects, 0, objects.size());
    
    printf("[BVH] Build complete. Total nodes: %lu\n", nodes.size());
}

int BVH::build_recursive(std::vector<Object>& objects, int start, int end) {
    // 1. 创建新节点
    // 此时 nodes.size() 就是当前新节点的索引
    int node_idx = (int)nodes.size();
    nodes.push_back({}); // 先占个位，稍后填充数据
    
    // 引用不能在 push_back 后长期持有，因为 vector 扩容会导致失效
    // 所以我们下面用索引访问 nodes[node_idx]

    // 2. 计算当前所有物体的总包围盒与质心包围盒
    AABB bounds = AABB::empty();
    AABB centroid_bounds = AABB::empty();
    for (int i = start; i < end; i++) {
        bounds.grow(get_object_bounds(objects[i]));
        centroid_bounds.grow(get_centroid(objects[i]));
    }
    nodes[node_idx].bounds = bounds;

    int n_objs = end - start;

    // 3. 递归终止条件 (叶子节点)
    // 如果物体数不超过叶子阈值，就直接收成叶子。
    // 这样可以降低 BVH 深度和遍历时的分支开销。
    if (n_objs <= kLeafPrimitiveCount) {
        nodes[node_idx].is_leaf = 1;
        nodes[node_idx].primitive_offset = start;
        nodes[node_idx].primitive_count = n_objs;
        return node_idx;
    }

    // 4. 使用分桶 SAH 选择分裂轴和分裂位置
    Vec centroid_size = centroid_bounds.max - centroid_bounds.min;
    int best_axis = 0;
    float best_extent = centroid_size.x;
    if (centroid_size.y > best_extent) {
        best_axis = 1;
        best_extent = centroid_size.y;
    }
    if (centroid_size.z > best_extent) {
        best_axis = 2;
        best_extent = centroid_size.z;
    }
    nodes[node_idx].axis = best_axis;

    int mid = start + n_objs / 2;
    if (best_extent > 1e-6f) {
        SahBin bins[kSahBinCount];
        const float min_centroid = axis_value(centroid_bounds.min, best_axis);
        const float inv_extent = 1.0f / best_extent;

        for (int i = start; i < end; ++i) {
            const Vec centroid = get_centroid(objects[i]);
            float offset = (axis_value(centroid, best_axis) - min_centroid) * inv_extent;
            int bin_idx = std::min(kSahBinCount - 1, std::max(0, static_cast<int>(offset * kSahBinCount)));
            bins[bin_idx].count++;
            bins[bin_idx].bounds.grow(get_object_bounds(objects[i]));
        }

        AABB left_bounds[kSahBinCount - 1];
        AABB right_bounds[kSahBinCount - 1];
        int left_counts[kSahBinCount - 1];
        int right_counts[kSahBinCount - 1];

        AABB prefix_bounds = AABB::empty();
        int prefix_count = 0;
        for (int i = 0; i < kSahBinCount - 1; ++i) {
            prefix_count += bins[i].count;
            prefix_bounds.grow(bins[i].bounds);
            left_bounds[i] = prefix_bounds;
            left_counts[i] = prefix_count;
        }

        AABB suffix_bounds = AABB::empty();
        int suffix_count = 0;
        for (int i = kSahBinCount - 1; i >= 1; --i) {
            suffix_count += bins[i].count;
            suffix_bounds.grow(bins[i].bounds);
            right_bounds[i - 1] = suffix_bounds;
            right_counts[i - 1] = suffix_count;
        }

        float best_cost = 1e30f;
        int best_split_bin = -1;
        const float leaf_cost = surface_area(bounds) * static_cast<float>(n_objs);
        for (int i = 0; i < kSahBinCount - 1; ++i) {
            if (left_counts[i] == 0 || right_counts[i] == 0) {
                continue;
            }
            float cost =
                surface_area(left_bounds[i]) * static_cast<float>(left_counts[i]) +
                surface_area(right_bounds[i]) * static_cast<float>(right_counts[i]);
            if (cost < best_cost) {
                best_cost = cost;
                best_split_bin = i;
            }
        }

        // 小节点在 SAH 没有明显优势时，直接收成叶子可以减少遍历层数。
        if (best_split_bin >= 0 && !(n_objs <= 8 && best_cost >= leaf_cost)) {
            auto split_it = std::partition(objects.begin() + start,
                                           objects.begin() + end,
                                           [&](const Object& obj) {
                                               Vec centroid = get_centroid(obj);
                                               float offset =
                                                   (axis_value(centroid, best_axis) - min_centroid) * inv_extent;
                                               int bin_idx = std::min(kSahBinCount - 1,
                                                                      std::max(0, static_cast<int>(offset * kSahBinCount)));
                                               return bin_idx <= best_split_bin;
                                           });
            mid = static_cast<int>(split_it - objects.begin());
        }
    }

    // SAH 分裂失败或退化时，回退到排序后二分，保证树一定可构建。
    if (mid == start || mid == end) {
        auto comparator = [best_axis](const Object& a, const Object& b) {
            Vec ca = get_centroid(a);
            Vec cb = get_centroid(b);
            return axis_value(ca, best_axis) < axis_value(cb, best_axis);
        };
        std::sort(objects.begin() + start, objects.begin() + end, comparator);
        mid = start + n_objs / 2;
    }

    // 7. 递归构建子树
    // 这是一个"深度优先"过程。
    // 左子树的所有节点会先被 push 到 nodes 数组里。
    // 所以 left_child_idx 通常就是 node_idx + 1。
    int left_idx = build_recursive(objects, start, mid);
    int right_idx = build_recursive(objects, mid, end);

    // 8. 填充内部节点数据
    // 注意：vector 可能扩容了，重新获取引用
    nodes[node_idx].is_leaf = 0; 
    nodes[node_idx].left_child_idx = left_idx; 
    nodes[node_idx].right_child_idx = right_idx; 

    return node_idx;
}
