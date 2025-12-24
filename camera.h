#pragma once
#include "common.h"
#include "scene.h" // 需要 CameraParams 定义

class CameraController {
public:
    // 构造函数：设置初始位置和朝向
    CameraController(Vec position, Vec direction);

    // 处理输入，返回 true 如果相机动了
    bool update(float delta_time);

    // 获取传给 GPU 的参数包
    CameraParams get_params(int width, int height);

    // [新增] 获取当前参数用于文件名
    float get_aperture() const { return aperture; }
    float get_focus_dist() const { return focus_dist; }

private:
    Vec pos;
    Vec dir;
    float speed = 2.5f;

    float aperture = 0.0f;     // 初始光圈 (0=无景深)
    float focus_dist = 240.0f; // 初始对焦距离 (大概在玻璃球的位置)
};
