#pragma once
#include "common.h"
#include "scene.h"

class CameraController {
public:
    CameraController(Vec position, Vec look_at);

    // [修改] update 现在只处理键盘移动
    // 返回 true 表示位置改变了
    bool update(float delta_time);

    // [新增] 处理鼠标移动
    // xrel, yrel: 鼠标在 X/Y 轴的相对位移
    // 返回 true 表示视角改变了
    bool process_mouse(float xrel, float yrel);

    CameraParams get_params(int width, int height);
    
    // [新增] 获取参数用于调试
    float get_aperture() const { return aperture; }
    float get_focus_dist() const { return focus_dist; }

private:
    void update_camera_vectors(); // 根据 Yaw/Pitch 计算 dir, right, up

    Vec pos;
    Vec dir;   // 前方
    Vec right; // 右方 (用于 A/D 移动)
    Vec up;    // 上方 (用于 Q/E 移动)

    // 欧拉角
    float yaw = -90.0f; // 初始看向 -Z 方向
    float pitch = 0.0f;
    
    float move_speed = 2.5f;
    float mouse_sensitivity = 0.1f; // 鼠标灵敏度

    float aperture = 0.0f;
    float focus_dist = 240.0f;
};
