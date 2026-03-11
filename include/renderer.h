#pragma once
#include "common.h"
#include "scene.h"
#include "bvh.h"
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>
#include <string>

/**
 * @file renderer.h
 * @brief SYCL 渲染后端接口定义
 * 
 * 职责:
 * 1. 管理 GPU 资源 (USM 内存分配与释放)。
 * 2. 处理场景数据的上传与转换。
 * 3. 发射渲染内核并处理同步。
 */

// 渲染器初始化逻辑
void init_renderer_sycl();

/// 返回渲染器内部实际使用的 SYCL 队列，便于主程序与内核共享同一上下文。
sycl::queue& get_renderer_queue();

/**
 * @brief 路径追踪诊断统计
 *
 * 这些计数器的目标不是做最终性能监控，而是帮助定位 fireflies 的来源：
 * 1. 直接光贡献是否经常爆亮；
 * 2. throughput 是否在某个 BSDF 分支中失控；
 * 3. 最终颜色是否仍然经常触发兜底 firefly clamp。
 */
struct RenderStats {
    uint32_t nan_or_inf_pixels = 0;
    uint32_t direct_light_clamp_count = 0;
    uint32_t emissive_hit_clamp_count = 0;
    uint32_t emissive_hit_primary_count = 0;
    uint32_t emissive_hit_indirect_count = 0;
    uint32_t emissive_hit_primary_clamp_count = 0;
    uint32_t emissive_hit_indirect_clamp_count = 0;
    uint32_t throughput_clamp_diffuse_count = 0;
    uint32_t throughput_clamp_specular_count = 0;
    uint32_t throughput_clamp_refract_count = 0;
    uint32_t throughput_clamp_rr_count = 0;
    uint32_t final_firefly_clamp_count = 0;

    uint32_t max_direct_light_lum_milli = 0;
    uint32_t max_emissive_hit_lum_milli = 0;
    uint32_t max_emissive_hit_primary_lum_milli = 0;
    uint32_t max_emissive_hit_indirect_lum_milli = 0;
    uint32_t max_throughput_component_milli = 0;
    uint32_t max_final_color_lum_milli = 0;
};

/**
 * @brief 初始化并上传场景数据
 * 
 * @param objects 场景中的几何物体列表
 * @param texture_files 纹理路径列表 (目前由 init_scene_data 内部管理加载逻辑)
 * @param nodes 预构建好的线性 BVH 节点数组
 * @param light_indices 光源索引列表 (用于 NEE 采样)
 */
void init_scene_data(const std::vector<Object>& objects, 
                     const std::vector<std::string>& texture_files,
                     const std::vector<LinearBVHNode>& nodes,
                     const std::vector<int>& light_indices);

/**
 * @brief 启动路径追踪内核
 * 
 * @param accum_buffer_usm 累加缓冲区 (必须是 USM Shared 类型)
 * @param width 画面宽度
 * @param height 画面高度
 * @param frame_seed 随机数种子 (通常使用当前帧序号)
 * @param tx 工作组宽度 (建议针对 Intel GPU 设为 16)
 * @param ty 工作组高度 (建议针对 Intel GPU 设为 8)
 * @param cam 相机参数 (位置、视角等)
 * @param stats_usm 诊断统计缓冲区 (必须是 USM Shared 类型，可为空)
 */
void launch_render_kernel(Vec* accum_buffer_usm,
                          int width,
                          int height,
                          int frame_seed,
                          int tx,
                          int ty,
                          CameraParams cam,
                          RenderStats* stats_usm);
