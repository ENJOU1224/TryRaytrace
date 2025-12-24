#include "input.h"

InputManager::InputManager() {
    // 初始化时默认锁定鼠标
    SDL_SetRelativeMouseMode(SDL_TRUE);
}

void InputManager::toggle_mouse_lock() {
    mouse_locked = !mouse_locked;
    SDL_SetRelativeMouseMode(mouse_locked ? SDL_TRUE : SDL_FALSE);
}

InputState InputManager::process_events(CameraController& cam) {
    InputState state; // 默认全是 false
    SDL_Event e;

    // 1. 处理离散事件 (按键按下、鼠标移动)
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            state.quit = true;
        }
        
        // 鼠标移动 (仅在锁定模式下生效)
        if (e.type == SDL_MOUSEMOTION && mouse_locked) {
            if (cam.process_mouse(e.motion.xrel, e.motion.yrel)) {
                state.camera_moved = true;
            }
        }

        // 键盘单次触发
        if (e.type == SDL_KEYDOWN) {
            switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: state.quit = true; break;
                case SDLK_p:      state.save_request = true; break;
                case SDLK_TAB:    toggle_mouse_lock(); break;
            }
        }
        
        // 鼠标点击 (如果未锁定，点击窗口重新锁定)
        if (e.type == SDL_MOUSEBUTTONDOWN && !mouse_locked) {
            toggle_mouse_lock();
        }
    }

    // 2. 处理连续输入 (WASD 移动)
    // 传入 1.0f 作为简单的 delta time
    if (cam.update(1.0f)) {
        state.camera_moved = true;
    }

    return state;
}
