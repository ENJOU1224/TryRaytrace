#pragma once
#include "common.h"
#include "scene.h"
#include "aabb.h"
#include <vector>

// BVH = Bounding Volume Hierarchy，中文常译作“包围体层次结构”。
// 它的目标很直接：
// “不要让每一条光线都和场景里所有三角形硬碰硬。”
//
// 做法是把空间递归切成二叉树：
// - 内部节点只存一个较大的包围盒和两个孩子
// - 叶子节点存少量真正的三角形
//
// GPU 上更偏好“线性数组”而不是一堆指针节点，所以这里用 LinearBVHNode。

// ======================================================================================
// 线性 BVH 节点 (GPU Friendly)
// ======================================================================================
// 为了 GPU 读取效率，我们需要紧凑的数据结构。
// 这里的节点既可以是"内部节点"(包含左右子树)，也可以是"叶子节点"(包含物体)。
struct ALIGN(16) LinearBVHNode {
    AABB bounds; 

    union {
        // 内部节点时使用：左孩子节点在线性数组中的下标
        int left_child_idx; 
        // 叶子节点时使用：该叶子里第一号三角形在 objects 数组中的起始位置
        int primitive_offset; 
    };

    union {
        // 内部节点时使用：右孩子节点在线性数组中的下标
        int right_child_idx; 
        // 叶子节点时使用：该叶子中一共存了多少个三角形
        int primitive_count; 
    };
    
    // 记录本节点主要沿哪个轴做切分，便于调试和后续可能的遍历优化。
    int axis;    
    // 1 表示叶子节点，0 表示内部节点。
    int is_leaf; 
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
    //            注意这是 C++ 常见的“左闭右开区间”，end 本身不包含在内。
    // 返回: 当前节点在 nodes 数组中的索引
    int build_recursive(std::vector<Object>& objects, int start, int end);
};
