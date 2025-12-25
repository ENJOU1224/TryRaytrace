#pragma once
#include "scene.h" // 需要 Object, Vec, Refl_t 的定义
#include <vector>

// OBJ 加载函数
// filename: 文件路径
// objects:  目标容器 (会把解析出的三角形追加到这里)
// offset:   位置偏移
// scale:    缩放倍数
// color:    材质颜色
// refl:     材质类型
void load_obj(const char* filename, std::vector<Object>& objects, 
              Vec offset, float scale, Vec color, Refl_t refl, float fuzz = 0.0f);
