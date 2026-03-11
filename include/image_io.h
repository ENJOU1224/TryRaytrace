#pragma once

#include "common.h"

/**
 * @file image_io.h
 * @brief 图像与日志输出接口
 */

/**
 * @brief 将当前累积缓冲区保存为快照，并在 `logs/` 下附带一份简单文本日志
 *
 * @param h_accum      当前累积颜色缓冲区
 * @param w            图像宽度
 * @param h            图像高度
 * @param frame        当前累积帧数
 * @param focus_dist   当前相机焦距
 * @param aperture     当前相机光圈
 */
void save_snapshot(const Vec* h_accum, int w, int h, int frame, float focus_dist, float aperture);
