#include "camera.h"
#include <SDL2/SDL.h> // 需要处理键盘输入
#include <iostream>

CameraController::CameraController(Vec position, Vec direction) 
    : pos(position), dir(direction) {}

bool CameraController::update(float delta_time) {
    bool moved = false;
    const Uint8* state = SDL_GetKeyboardState(NULL);
    
    // 简单的 WASD 移动
    // 这里为了演示简单，直接修改 pos
    // 实际引擎中应该根据 dir 向量来计算前/后/左/右
    if (state[SDL_SCANCODE_W]) { pos.z -= speed; moved = true; }
    if (state[SDL_SCANCODE_S]) { pos.z += speed; moved = true; }
    if (state[SDL_SCANCODE_A]) { pos.x -= speed; moved = true; }
    if (state[SDL_SCANCODE_D]) { pos.x += speed; moved = true; }
    if (state[SDL_SCANCODE_Q]) { pos.y += speed; moved = true; }
    if (state[SDL_SCANCODE_E]) { pos.y -= speed; moved = true; }

    // R/F: 调整对焦距离 (Focus Distance)
    if (state[SDL_SCANCODE_R]) { 
        focus_dist += 1.0f; 
        moved = true; 
        std::cout << "Focus Dist: " << focus_dist << std::endl; 
    }
    if (state[SDL_SCANCODE_F]) { 
        focus_dist -= 1.0f; 
        if (focus_dist < 1.0f) focus_dist = 1.0f; // 防止负数
        moved = true; 
        std::cout << "Focus Dist: " << focus_dist << std::endl; 
    }
    
    // T/G: 调整光圈大小 (Aperture)
    if (state[SDL_SCANCODE_T]) { 
        aperture += 0.1f; 
        moved = true; 
        std::cout << "Aperture: " << aperture << std::endl; 
    }
    if (state[SDL_SCANCODE_G]) { 
        aperture -= 0.1f; 
        if (aperture < 0.0f) aperture = 0.0f; 
        moved = true; 
        std::cout << "Aperture: " << aperture << std::endl; 
    }

    return moved;
}

CameraParams CameraController::get_params(int width, int height) {
    Vec n_dir = dir.norm();
    // 根据长宽比计算视口
    Vec cx = {width * .5135f / height, 0, 0};
    Vec cy = cx.cross(n_dir).norm() * .5135f;
    
    // [修改] 传入 lens_radius (光圈直径的一半) 和 focus_dist
    return {pos, cx, cy, n_dir, aperture / 2.0f, focus_dist};
}
