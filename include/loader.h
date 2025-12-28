#pragma once
#include "scene.h" // 需要 Object, Vec, Refl_t 的定义
#include <vector>

// OBJ 加载函数
// filename:      文件路径
// objects:       目标容器 (会把解析出的三角形追加到这里)
// offset:        位置偏移
// scale:         缩放倍数
// rotation:      旋转 
// albedo:        基本色 
// metallic:      金属性
// roughness:     粗糙度
// transmission:  透明度
// ior:           折射率
// tex_id:        纹理编号
// smooth:        平滑 
void load_obj(const char* filename, std::vector<Object>& objects, 
              Vec offset, float scale, Vec rotation, // [新增] rotation
              Vec albedo, float metallic, float roughness, float transmission = 0.0f, float ior = 1.45f,
              int tex_id = -1, bool smooth = true);

// ppm 图片加载函数
// filename:  文件路径
// w:         宽度
// h:         高度
unsigned char* load_ppm(const char* filename, int* w, int* h);
