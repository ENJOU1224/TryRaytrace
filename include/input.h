#pragma once
#include <SDL2/SDL.h>
#include "camera.h" // 需要操作相机

struct InputState {
    bool quit = false;          // 是否退出程序
    bool save_request = false;  // 是否请求保存图片
    bool camera_moved = false;  // 相机是否发生移动(需要重置渲染)
};

class InputManager {
public:
    InputManager();
    
    // 处理一帧内的所有事件
    // 返回: InputState 结构体，告诉主程序该干嘛
    InputState process_events(CameraController& cam);

private:
    void toggle_mouse_lock();
    bool mouse_locked = true; // 默认锁定
};
