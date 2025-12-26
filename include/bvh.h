#pragma once
#include "common.h"
#include "scene.h"
#include "aabb.h"
#include <vector>

// ======================================================================================
// 线性 BVH 节点 (GPU Friendly)
// ======================================================================================
// 为了 GPU 读取效率，我们需要紧凑的数据结构。
// 这里的节点既可以是"内部节点"(包含左右子树)，也可以是"叶子节点"(包含物体)。
struct __align__(16) LinearBVHNode {
    AABB bounds; // 包围盒 (32 bytes: 16 min + 16 max)

    // 联合体
    union {
        int left_child_idx; // 内部节点用: 左孩子索引
        int primitive_offset; // 叶子节点用: 物体在数组中的偏移
    };

    union {
        int right_child_idx; // 内部节点用: 右孩子索引
        int primitive_count; // 叶子节点用: 物体数量
    };
    
    int axis;    // 调试用
    int is_leaf; // 1=Leaf, 0=Internal (单独一个标记位最清晰)
};

// ======================================================================================
// BVH 构建器
// ======================================================================================
class BVH {
public:
    // 构建函数
    // 注意: 这会重新排序 objects 数组！
    // 因为 BVH 要求叶子节点里的物体在内存中必须是连续的。
    void build(std::vector<Object>& objects);

    // 获取构建好的节点数组 (传给 GPU)
    const std::vector<LinearBVHNode>& get_nodes() const { return nodes; }

private:
    std::vector<LinearBVHNode> nodes;

    // 递归构建函数
    // objects: 物体列表
    // start, end: 当前节点覆盖的物体范围 [start, end)
    // 返回: 当前节点在 nodes 数组中的索引
    int build_recursive(std::vector<Object>& objects, int start, int end);
};
