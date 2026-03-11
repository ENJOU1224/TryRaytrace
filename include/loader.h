#pragma once

#include "scene.h"

#include <vector>

/**
 * @file loader.h
 * @brief Wavefront OBJ 加载接口
 *
 * 当前加载器只处理最常见、最简单的三角面 OBJ：
 * - 顶点行: `v x y z`
 * - 面行:   `f i j k`
 *
 * 加载后会直接生成世界空间三角形，并在 CPU 端完成几何预计算。
 */

/**
 * @brief 读取 OBJ 并把三角形追加到场景对象数组中
 *
 * @param filename OBJ 文件路径
 * @param objects  目标容器，解析出的三角形会 append 到这里
 * @param offset   模型整体平移
 * @param scale    模型整体缩放
 * @param albedo   材质基础色
 * @param metallic 金属度
 * @param roughness 粗糙度
 */
void load_obj(const char* filename,
              std::vector<Object>& objects,
              Vec offset,
              float scale,
              Vec albedo,
              float metallic,
              float roughness);
