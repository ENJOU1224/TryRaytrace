#include "camera.h"
#include <SDL2/SDL.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio> // for printf

// ======================================================================================
// 数学辅助函数
// ======================================================================================
// 将角度 (Degrees) 转换为 弧度 (Radians)
// C++ 标准库的三角函数 (sin, cos) 接收的参数都是弧度。
inline float radians(float deg) { 
    return deg * (M_PI / 180.0f); 
}

// ======================================================================================
// 构造函数
// ======================================================================================
CameraController::CameraController(Vec position, Vec look_at) 
    : pos(position) {
    // 这里的 look_at 参数在目前的第一人称漫游模式下暂未使用。
    // 我们默认通过 yaw/pitch = -90/0 来初始化朝向 (看向 -Z)。
    // 如果需要"看向特定点"的功能，需要在这里用 atan2 计算初始 yaw/pitch。
    
    // 初始化相机的局部坐标系 (Front, Right, Up)
    update_camera_vectors();
}

// ======================================================================================
// 核心数学: 更新相机向量 (Euler Angles -> Vectors)
// ======================================================================================
// [数学原理] 球坐标系转笛卡尔坐标系
// 根据 Yaw (偏航角) 和 Pitch (俯仰角) 计算出单位方向向量。
// 
// 1. Pitch (φ): 决定了 Y 分量 (sinφ) 和 XZ 平面的投影长度 (cosφ)。
// 2. Yaw (θ): 决定了 XZ 平面上 X (cosθ) 和 Z (sinθ) 的分配。
void CameraController::update_camera_vectors() {
    Vec front;
    
    // 1. 计算指向前方的向量 (Front)
    // 注意: 这里的三角函数组合是标准的球坐标转换公式
    front.x = cos(radians(yaw)) * cos(radians(pitch));
    front.y = sin(radians(pitch));
    front.z = sin(radians(yaw)) * cos(radians(pitch));
    
    // 归一化，确保长度为 1
    dir = front.norm();

    // 2. 计算 Right 和 Up 向量 (构建正交基底)
    // Gram-Schmidt 正交化过程的简化版
    
    // 假设世界坐标系的"上"永远是 Y 轴 (0, 1, 0)
    static const Vec world_up = {0, 1, 0};
    
    // Right = Front × WorldUp (利用叉积算出垂直于这两个向量的向量)
    right = dir.cross(world_up).norm();
    
    // Up = Right × Front (算出垂直于 Right 和 Front 的向量，即相机的头顶方向)
    up = right.cross(dir).norm();
}

// ======================================================================================
// 输入处理: 鼠标旋转
// ======================================================================================
bool CameraController::process_mouse(float xrel, float yrel) {
    // 1. 根据鼠标位移更新欧拉角
    yaw   += xrel * mouse_sensitivity;
    pitch -= yrel * mouse_sensitivity; // 注意: 鼠标向上推(y减小)通常对应抬头(pitch增加)，这里符号根据习惯调整

    // 2. 限制俯仰角 (Gimbal Lock Prevention)
    // 防止头抬得太高导致翻转，通常限制在 +/- 89 度
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    // 3. 角度变了，必须重新计算向量
    update_camera_vectors();
    
    return true; // 返回 true 告诉主程序: 相机动了，请重置渲染累积
}

// ======================================================================================
// 输入处理: 键盘移动 & 参数调整
// ======================================================================================
bool CameraController::update(float delta_time) {
    bool moved = false;
    const Uint8* state = SDL_GetKeyboardState(NULL);
    
    // [物理位移]
    // 速度 = 基础速度 * 时间增量
    // 注意: 目前 main.cpp 传入的 delta_time 是 1.0f (按帧移动)，如果改为真实时间(dt)，移动会更平滑
    float velocity = move_speed * delta_time; 

    // W/S: 前后移动 (沿着相机的观察方向 dir)
    if (state[SDL_SCANCODE_W]) { pos = pos + dir * velocity; moved = true; }
    if (state[SDL_SCANCODE_S]) { pos = pos - dir * velocity; moved = true; }
    
    // A/D: 左右平移 (沿着相机的右侧方向 right)
    if (state[SDL_SCANCODE_A]) { pos = pos - right * velocity; moved = true; }
    if (state[SDL_SCANCODE_D]) { pos = pos + right * velocity; moved = true; }
    
    // Q/E: 垂直升降 (简单起见，这里沿世界 Y 轴升降，也可以改为沿相机 up 向量升降)
    // 沿世界 Y 轴: 像坐电梯
    // 沿相机 up 轴: 像飞机爬升
    if (state[SDL_SCANCODE_Q]) { pos.y += velocity; moved = true; }
    if (state[SDL_SCANCODE_E]) { pos.y -= velocity; moved = true; }

    // [参数热调节]
    // R/F: 调节对焦距离 (Focus Distance)
    if (state[SDL_SCANCODE_R]) { 
        focus_dist += 1.0f; moved = true; 
        printf("[Cam] Focus: %.1f\n", focus_dist); 
    }
    if (state[SDL_SCANCODE_F]) { 
        focus_dist -= 1.0f; 
        if(focus_dist < 1.0f) focus_dist = 1.0f; 
        moved = true; 
        printf("[Cam] Focus: %.1f\n", focus_dist); 
    }
    
    // T/G: 调节光圈大小 (Aperture)
    if (state[SDL_SCANCODE_T]) { 
        aperture += 0.1f; moved = true; 
        printf("[Cam] Aperture: %.1f\n", aperture); 
    }
    if (state[SDL_SCANCODE_G]) { 
        aperture -= 0.1f; 
        if(aperture < 0.0f) aperture = 0.0f; 
        moved = true; 
        printf("[Cam] Aperture: %.1f\n", aperture); 
    }

    return moved;
}

// ======================================================================================
// 数据打包: 生成 GPU 需要的参数
// ======================================================================================
CameraParams CameraController::get_params(int width, int height) {
    // [视口计算 (Viewport Calculation)]
    // 我们需要把相机的"朝向"转换成成像平面的"两个轴" (cx, cy)。
    // 这决定了视野范围 (FOV) 和 画面比例 (Aspect Ratio)。

    // 1. FOV 系数
    // 0.5135 是 tan(FOV / 2) 的值。
    // 对应垂直 FOV 约为 54.4 度 (2 * atan(0.5135))。
    const float fov_scale = 0.5135f; 
    
    // 2. 纵横比 (Aspect Ratio)
    float aspect = (float)width / height;

    // 3. 计算成像平面的基向量
    // 我们利用之前算好的 right 和 up 向量，因为它们天然垂直于视线。
    // cx (水平轴): 方向是 right，长度取决于 纵横比 * FOV
    // cy (垂直轴): 方向是 up，   长度取决于 FOV
    Vec cx = right * (fov_scale * aspect);
    Vec cy = up * fov_scale;
    
    // 4. 打包返回
    // 这里的 dir 已经是归一化的，可以直接给 GPU 用
    // aperture * 0.5f: GPU 也是用半径计算，这里把直径除以2
    return {pos, cx, cy, dir, aperture * 0.5f, focus_dist};
}
