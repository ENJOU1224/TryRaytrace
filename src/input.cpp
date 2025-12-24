#include "input.h"
// SDL2 头文件已经在 input.h 中包含，这里不需要重复，
// 但为了显式依赖关系，保留也无妨。

// ======================================================================================
// 构造函数
// ======================================================================================
InputManager::InputManager() {
    // [初始化状态]
    // 默认开启"相对鼠标模式" (Relative Mouse Mode)。
    // 效果: 
    // 1. 隐藏系统光标。
    // 2. 锁定光标在窗口中心，无法移出边界。
    // 3. 鼠标移动报告的是"相对位移" (delta x, delta y)，而非屏幕绝对坐标。
    //这是 FPS 游戏漫游的标准模式。
    SDL_SetRelativeMouseMode(SDL_TRUE);
    mouse_locked = true;
}

// ======================================================================================
// 切换鼠标锁定状态
// ======================================================================================
void InputManager::toggle_mouse_lock() {
    mouse_locked = !mouse_locked;
    
    // 同步更新 SDL 的状态
    if (mouse_locked) {
        SDL_SetRelativeMouseMode(SDL_TRUE); // 锁定 & 隐藏
    } else {
        SDL_SetRelativeMouseMode(SDL_FALSE); // 解锁 & 显示
        // 解锁后，用户可以把鼠标移出窗口去操作其他软件
    }
}

// ======================================================================================
// 核心：处理一帧内的所有输入
// ======================================================================================
// 参数: 
//   cam: 相机控制器引用 (我们需要直接修改相机的状态)
// 返回: 
//   InputState: 包含"是否退出"、"是否保存"、"相机是否移动"等标志位，供 Main 使用。
// ======================================================================================
InputState InputManager::process_events(CameraController& cam) {
    InputState state; // 初始化状态 (bool 默认为 false)
    SDL_Event e;

    // ------------------------------------------------------------------
    // 1. 事件轮询 (Event Polling)
    // ------------------------------------------------------------------
    // SDL 会把键盘、鼠标、窗口系统的所有事件放入队列。
    // 我们必须在一个循环里把它们全部处理完，否则程序会"卡死"或无响应。
    while (SDL_PollEvent(&e)) {
        
        // [窗口关闭事件] (点击右上角 X)
        if (e.type == SDL_QUIT) {
            state.quit = true;
        }
        
        // [鼠标移动事件]
        if (e.type == SDL_MOUSEMOTION) {
            // 关键逻辑: 只有在"锁定模式"下，鼠标移动才控制相机旋转。
            // 如果用户按 Tab 解锁了鼠标，这里的移动应当被忽略，否则画面会乱晃。
            if (mouse_locked) {
                // xrel, yrel 是相对位移
                if (cam.process_mouse(e.motion.xrel, e.motion.yrel)) {
                    state.camera_moved = true; // 标记相机动了 -> 通知 Main 重置累加器
                }
            }
        }

        // [键盘按键事件 (单次触发)]
        // 适用于开关型操作 (Toggle) 或 单次指令 (Command)
        if (e.type == SDL_KEYDOWN) {
            switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: 
                    state.quit = true; 
                    break;
                case SDLK_p:      
                    state.save_request = true; // 请求保存截图
                    break;
                case SDLK_TAB:    
                    toggle_mouse_lock(); // 切换鼠标锁定
                    break;
            }
        }
        
        // [鼠标点击事件]
        // 用户体验优化: 如果鼠标不小心解锁了，点击窗口任意位置重新锁定
        if (e.type == SDL_MOUSEBUTTONDOWN) {
            if (!mouse_locked) {
                toggle_mouse_lock();
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. 连续状态检测 (Continuous State)
    // ------------------------------------------------------------------
    // 对于 WASD 移动，我们不使用 SDL_KEYDOWN 事件。
    // 因为事件依赖系统的"按键重复率"，会有延迟和卡顿。
    // 我们直接查询键盘的"当前状态" (Snapshot)，只要按住，每一帧都会触发移动。
    
    // 参数 1.0f 是时间步长 (Delta Time)。
    // 在更完善的引擎中，这里应该传入 (current_time - last_time)，实现帧率无关的移动速度。
    if (cam.update(1.0f)) {
        state.camera_moved = true;
    }

    return state;
}
