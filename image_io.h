#pragma once
#include "common.h"

// 保存快照到 log 文件夹
// 参数: 累加缓冲区数据, 宽, 高, 当前帧数, (可选的元数据: 焦距, 光圈)
void save_snapshot(const Vec* h_accum, int w, int h, int frame, float focus_dist, float aperture);
