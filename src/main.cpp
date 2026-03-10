#include <csignal>
#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>

#include "common.h"     
#include "scene.h"      
#include "camera.h"     
#include "renderer.h"   
#include "input.h"      
#include "image_io.h"   
#include "bvh.h"        

/**
 * @file main.cpp
 * @brief 系统主入口与调度器
 * 
 * 职责:
 * 1. 资源管理: 初始化 SDL2 窗口、SYCL 队列及 USM 共享内存。
 * 2. 场景驱动: 调用各模块完成场景构建、BVH 划分及数据上传。
 * 3. 实时渲染循环: 协调 GPU 计算、CPU 后处理与屏幕显示。
 */

std::atomic<bool> quit(false);
void signal_handler(int signal) { if (signal == SIGINT) quit = true; }

int main(int argc, char** argv) {
    // 注册信号处理，支持 Ctrl+C 安全退出
    std::signal(SIGINT, signal_handler);

    // 1. 系统参数配置
    const int w = 1200;
    const int h = 800;  
    
    // 初始化 SDL2 视频子系统
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[SDL Error] Initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    // 创建图形窗口与渲染上下文
    SDL_Window* window = SDL_CreateWindow("Lunar-Raytrace (oneAPI/SYCL)", 
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                          w, h, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, 
                                             SDL_TEXTUREACCESS_STREAMING, w, h);

    // 2. 场景与渲染引擎初始化
    // 创建场景数据 (Cornell Box + OBJ Models)
    Scene scene = create_cornell_box();
    
    // 构建 BVH 加速结构 (线性化处理)
    BVH bvh; 
    bvh.build(scene.objects);

    // 筛选光源索引 (用于 NEE 显式采样优化)
    std::vector<int> light_indices;
    for (size_t i = 0; i < scene.objects.size(); i++) {
        if (scene.objects[i].emission.norm_len() > 0.1f) {
            light_indices.push_back((int)i);
        }
    }

    // 初始化 SYCL 渲染核心并上传数据到 USM (统一内存)
    init_scene_data(scene.objects, scene.texture_files, bvh.get_nodes(), light_indices);

    // 初始化相机控制器 (第一人称视角)
    CameraController cam({50, 50, 295.6}, {0, 0, -1});

    // 3. 运行时资源准备
    // 自动选择最佳 SYCL 设备 (优先 Level Zero GPU)
    sycl::queue q;
    try { 
        q = sycl::queue(sycl::gpu_selector_v); 
    } catch (...) { 
        q = sycl::queue(sycl::cpu_selector_v); 
    }

    // 分配 USM Shared Memory：CPU 与 GPU 物理共享，零拷贝访问
    Vec* d_accum = sycl::malloc_shared<Vec>(w * h, q);
    std::memset(d_accum, 0, w * h * sizeof(Vec));
    
    // CPU 侧像素映射缓冲区 (用于 SDL 显示)
    uint32_t* pixel_buffer = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    InputManager input; 
    int gpu_frame = 1;
    
    // FPS 与性能统计
    auto last_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;

    // ------------------------------------------------------------------
    // 4. 实时渲染主循环
    // ------------------------------------------------------------------
    while (!quit) {
        // [A] 处理用户输入
        InputState state = input.process_events(cam);
        
        // 保存当前快照
        if (state.save_request) {
            save_snapshot(d_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture()); 
        }
        if (state.quit) quit = true;

        // 相机移动时重置累加器，确保画面不产生拖影
        if (state.camera_moved) {
            gpu_frame = 1;
            std::memset(d_accum, 0, w * h * sizeof(Vec));
        }

        // [B] 发射渲染内核
        CameraParams cam_params = cam.get_params(w, h);
        // 建议 tx=16, ty=8 以匹配 Intel Xe2 硬件 Sub-group 布局
        launch_render_kernel(d_accum, w, h, gpu_frame, 16, 8, cam_params);
        
        // [C] 图像后处理 (Tone Mapping & Gamma Correction)
        // 利用 OpenMP 并行化处理 100万+ 个像素的指数运算
        #pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
            Vec color = d_accum[i] * (1.0f / gpu_frame);
            pixel_buffer[i] = (255 << 24) | (toInt(color.x) << 16) | (toInt(color.y) << 8) | toInt(color.z);
        }

        // [D] 更新 SDL 屏幕
        SDL_UpdateTexture(texture, NULL, pixel_buffer, w * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        
        // [E] 性能统计与反馈
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = current_time - last_time;
        last_time = current_time;
        fps = 0.9f * fps + 0.1f * (1.0f / delta.count()); 

        if (gpu_frame % 5 == 0) {
            char title[256];
            sprintf(title, "[%s] FPS: %.1f | Frame: %d | F: %.1f | A: %.2f", 
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    fps, gpu_frame, cam.get_focus_dist(), cam.get_aperture());
            SDL_SetWindowTitle(window, title);

            // 命令行进度刷新
            printf("\r>> [SYCL] Rendering: Frame %d | FPS: %.1f", gpu_frame, fps);
            fflush(stdout);
        }
        gpu_frame++;
    }

    // 退出前自动存档
    save_snapshot(d_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture()); 

    // 5. 资源清理
    sycl::free(d_accum, q); 
    free(pixel_buffer);
    SDL_DestroyTexture(texture); 
    SDL_DestroyRenderer(renderer); 
    SDL_DestroyWindow(window); 
    SDL_Quit();

    std::cout << "\n[System] Render session ended safely." << std::endl;
    return 0;
}
