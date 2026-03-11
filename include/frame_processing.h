#pragma once

#include "common.h"

#include <cstdint>
#include <vector>

/**
 * @file frame_processing.h
 * @brief 帧后处理辅助函数
 *
 * 这组函数只负责“把线性 HDR 颜色变成能显示/能送进降噪器的数据”。
 * 之所以单独拆出来，是为了把 main.cpp 从大段像素遍历代码里解放出来，
 * 让主循环更像“调度器”，而不是“所有细节都堆在一起的大函数”。
 */

/// 将累积缓冲区转换为线性 RGB 图像，供降噪器使用。
void build_linear_rgb_frame(const Vec* accum, int pixel_count, int frame_index, std::vector<float>& rgb_buffer);

/// 将线性 RGB 图像转换成 SDL 可直接显示的 ARGB8888。
void linear_rgb_to_argb8888(const float* rgb_buffer, int pixel_count, uint32_t* pixel_buffer);

/// 当关闭降噪时，直接从累积缓冲区生成显示图像，减少一次中间缓冲遍历。
void accum_to_argb8888(const Vec* accum, int pixel_count, int frame_index, uint32_t* pixel_buffer);
