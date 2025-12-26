#include "bvh.h"
#include <algorithm>
#include <iostream>

// 获取物体的中心点 (用于排序)
Vec get_centroid(const Object& obj) {
    // 三角形重心
    return (obj.v0 + obj.v1 + obj.v2) * 0.333333f;
}

// 获取物体的 AABB
AABB get_object_bounds(const Object& obj) {
    AABB box = AABB::empty();

    box.grow(obj.v0);
    box.grow(obj.v1);
    box.grow(obj.v2);

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

    // 2. 计算当前所有物体的总包围盒
    AABB bounds = AABB::empty();
    for (int i = start; i < end; i++) {
        bounds.grow(get_object_bounds(objects[i]));
    }
    nodes[node_idx].bounds = bounds;

    int n_objs = end - start;

    // 3. 递归终止条件 (叶子节点)
    // 如果只剩 1 个物体，这就必须是叶子了
    if (n_objs == 1) {
        nodes[node_idx].is_leaf = 1;
        nodes[node_idx].primitive_offset = start;
        nodes[node_idx].primitive_count = n_objs;
        return node_idx;
    }

    // 4. 寻找分裂轴 (Split Axis)
    // 选盒子最长的那条边切开
    Vec size = bounds.max - bounds.min;
    int axis = 0;
    if (size.y > size.x) axis = 1;
    if (size.z > size.y && size.z > size.x) axis = 2;
    nodes[node_idx].axis = axis;

    // 5. 排序 (Sorting)
    // 根据物体中心点在 split axis 上的坐标，把物体重新排列
    // std::sort 会真的改变 objects 数组的顺序！
    auto comparator = [axis](const Object& a, const Object& b) {
        Vec ca = get_centroid(a);
        Vec cb = get_centroid(b);
        if (axis == 0) return ca.x < cb.x;
        if (axis == 1) return ca.y < cb.y;
        return ca.z < cb.z;
    };
    
    std::sort(objects.begin() + start, objects.begin() + end, comparator);

    // 6. 分裂 (Split)
    // 简单的中点分裂：一半左边，一半右边
    int mid = start + n_objs / 2;

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
