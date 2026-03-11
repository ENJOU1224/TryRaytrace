#pragma once
#include <SDL2/SDL.h>
#include "camera.h" // 需要操作相机

// 输入模块的职责很单纯：
// 1. 把 SDL 的原始事件读出来；
// 2. 翻译成“主循环真正关心的几个布尔状态”；
// 3. 需要时直接驱动 CameraController 改变相机。

struct InputState {
    bool quit = false;          // 是否退出程序
    bool save_request = false;  // 是否请求保存图片
    bool camera_moved = false;  // 相机是否发生移动(需要重置渲染)
};

class InputManager {
public:
    InputManager();
    
    // 处理一帧内的所有事件
    // 返回: InputState 结构体，告诉主程序这一帧接下来该干嘛
    InputState process_events(CameraController& cam);

private:
    // SDL 的相对鼠标模式开关。
    // 锁定时更像 FPS 游戏；解锁时方便把鼠标移出窗口。
    void toggle_mouse_lock();
    bool mouse_locked = true; // 默认锁定
};
