#include "camera.h"
#include <SDL2/SDL.h>
#include <cmath>
#include <algorithm> // for std::clamp
#include <iostream>

// 辅助: 角度转弧度
inline float radians(float deg) { return deg * (M_PI / 180.0f); }

CameraController::CameraController(Vec position, Vec look_at) 
    : pos(position) {
    // 简单起见，这里不再从 look_at 反推角度，而是使用默认朝向
    // 初始化向量
    update_camera_vectors();
}

void CameraController::update_camera_vectors() {
    // 1. 根据欧拉角计算新的 Front 向量
    Vec front;
    front.x = cos(radians(yaw)) * cos(radians(pitch));
    front.y = sin(radians(pitch));
    front.z = sin(radians(yaw)) * cos(radians(pitch));
    dir = front.norm();

    // 2. 重新计算 Right 和 Up 向量
    // WorldUp 永远是 {0, 1, 0}
    Vec world_up = {0, 1, 0};
    right = dir.cross(world_up).norm();
    up    = right.cross(dir).norm();
}

bool CameraController::process_mouse(float xrel, float yrel) {
    // 1. 更新角度
    yaw   += xrel * mouse_sensitivity;
    pitch -= yrel * mouse_sensitivity; // 注意这里通常是减，因为鼠标往上(Y减小)是抬头

    // 2. 限制抬头/低头角度 (防止脖子折断)
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    // 3. 更新向量
    update_camera_vectors();
    
    return true; // 视角变了，需要重置渲染
}

bool CameraController::update(float delta_time) {
    bool moved = false;
    const Uint8* state = SDL_GetKeyboardState(NULL);
    float velocity = move_speed; // 简单起见忽略 delta_time

    // [修正] 现在的移动是相对于"视角"的
    // W/S: 沿着 dir 移动
    if (state[SDL_SCANCODE_W]) { pos = pos + dir * velocity; moved = true; }
    if (state[SDL_SCANCODE_S]) { pos = pos - dir * velocity; moved = true; }
    
    // A/D: 沿着 right 移动
    if (state[SDL_SCANCODE_A]) { pos = pos - right * velocity; moved = true; }
    if (state[SDL_SCANCODE_D]) { pos = pos + right * velocity; moved = true; }
    
    // Q/E: 垂直升降 (沿着世界坐标 Y 轴，或者用 up 向量)
    if (state[SDL_SCANCODE_Q]) { pos.y += velocity; moved = true; }
    if (state[SDL_SCANCODE_E]) { pos.y -= velocity; moved = true; }

    // 参数控制 (保持不变)
    if (state[SDL_SCANCODE_R]) { focus_dist += 1.0f; moved = true; printf("F: %.1f\n", focus_dist); }
    if (state[SDL_SCANCODE_F]) { focus_dist -= 1.0f; if(focus_dist<1) focus_dist=1; moved = true; printf("F: %.1f\n", focus_dist); }
    if (state[SDL_SCANCODE_T]) { aperture += 0.1f; moved = true; printf("A: %.1f\n", aperture); }
    if (state[SDL_SCANCODE_G]) { aperture -= 0.1f; if(aperture<0) aperture=0; moved = true; printf("A: %.1f\n", aperture); }

    return moved;
}

CameraParams CameraController::get_params(int width, int height) {
    // 这里的 dir 已经是 update_camera_vectors 计算好的
    Vec cx = {width * .5135f / height, 0, 0};
    // 注意: 原来的代码是用叉积算的 cy，现在我们可以更灵活
    // 为了保持原本的坐标系习惯，我们还是用叉积
    // 但必须确保 cx 是水平的。简单起见，我们重新构建相机基底
    
    // 实际上，为了支持任意旋转，我们需要更通用的 View Matrix 逻辑
    // 但为了兼容现有的 Renderer Kernel (它依赖 cx, cy 作为视口平面向量)
    // 我们需要构建一个垂直于 dir 的平面
    
    Vec cam_right = dir.cross({0,1,0}).norm();
    Vec cam_up    = cam_right.cross(dir).norm();
    
    // 重新调整视口向量的方向
    float aspect = (float)width / height;
    float fov_scale = .5135f; // 这里简化处理，保持之前的 FOV
    
    // cx 指向右边，cy 指向上边
    // 注意：原来的代码 cx 是 {w*... , 0, 0}，这是假设相机永远水平
    // 现在相机可以歪，所以 cx, cy 也要跟着旋转
    Vec final_cx = cam_right * (aspect * fov_scale * (float)height/width * width); // 有点绕，其实就是 cam_right * len
    // 简化: 
    // 原版: cx.x = width * .5135 / height.  长度 ≈ 0.5135 * Aspect
    float len_x = 0.5135f * aspect;
    float len_y = 0.5135f;
    
    return {pos, cam_right * len_x, cam_up * len_y, dir, aperture * 0.5f, focus_dist};
}
