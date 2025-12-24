/*
 * ======================================================================================
 * [主程序] 模块化 CUDA 光线追踪引擎 - 启动器 & 调度器
 * ======================================================================================
 * 
 * 职责:
 * 1. 资源管理: 初始化窗口、显存、内存、各个子模块 (Scene, Camera, Pipeline)。
 * 2. 主循环调度: 协调 GPU 计算与 CPU 后处理，实现"双缓冲流水线"并行。
 * 3. 交互逻辑: 处理用户输入 (WASD 漫游)，控制渲染状态 (累积/重置)。
 * 
 * 架构视角:
 * [Input] -> [Camera] -> [Renderer(GPU)] -> [VRAM Snapshot] -> [Pipeline(CPU)] -> [Display]
 */

#include <iostream>
#include <vector>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>

// 引入各个子模块的头文件
#include "common.h"     // 基础数据结构 (Vec)
#include "scene.h"      // 场景数据定义
#include "camera.h"     // 相机控制逻辑
#include "renderer.h"   // GPU 渲染接口
#include "pipeline.h"   // CPU 后台处理流水线
#include <csignal> // [新增] 信号处理库
#include <ctime>   // [新增] 时间库
#include <iomanip> // [新增] 格式化输出

// ======================================================================================
// 全局信号标志
// ======================================================================================
// volatile sig_atomic_t 保证在信号处理函数中修改变量是安全的
volatile sig_atomic_t g_signal_received = 0;

void handle_sigint(int sig) {
    g_signal_received = 1;
}

// ======================================================================================
// 快照保存函数
// ======================================================================================
void save_snapshot(Vec* h_accum, int w, int h, int frame, float focus, float aperture) {
    // 1. 获取当前时间字符串
    std::time_t now = std::time(nullptr);
    std::tm* t = std::localtime(&now);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", t);

    // 2. 生成文件名
    char filename[256];
    sprintf(filename, "log/%s_Frame%d_Focus%.1f_Aperture%.1f.ppm", 
            time_str, frame, focus, aperture);

    // 3. 准备数据 (从高精度累加数据转换)
    unsigned char* img = (unsigned char*)malloc(w * h * 3);
    
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        Vec avg = h_accum[i] * (1.0f / frame);
        img[i * 3 + 0] = (unsigned char)toInt(avg.x);
        img[i * 3 + 1] = (unsigned char)toInt(avg.y);
        img[i * 3 + 2] = (unsigned char)toInt(avg.z);
    }

    // 4. 写入文件
    FILE* f = fopen(filename, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
        fwrite(img, 1, w * h * 3, f);
        fclose(f);
        printf("\n[Snapshot] Image saved to: %s\n", filename);
    } else {
        printf("\n[Error] Failed to save snapshot!\n");
    }

    free(img);
}

// ======================================================================================
// 主函数入口
// ======================================================================================
int main(int argc, char** argv) {
    // [新增] 注册 Ctrl+C 信号处理
    signal(SIGINT, handle_sigint);

    // [新增] 创建 log 文件夹
    (void)system("mkdir -p log");

    // ------------------------------------------------------------------
    // 1. 系统配置
    // ------------------------------------------------------------------
    int w = 1200; // 窗口宽度
    int h = 800;  // 窗口高度
    
    // 初始化 SDL 视频子系统
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL 初始化失败: %s\n", SDL_GetError());
        return 1;
    }

    // 创建窗口 (Window)
    SDL_Window* window = SDL_CreateWindow(
        "CUDA RayTracing Engine (Modular)", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        w, h, 
        SDL_WINDOW_SHOWN
    );
    
    // 创建渲染器 (Renderer) - 这里指的是 SDL 的 2D 绘图器，利用硬件加速
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    
    // 创建纹理 (Texture) - 显存中的一块像素缓冲区，用于最终上屏显示
    // SDL_TEXTUREACCESS_STREAMING: 标记这块纹理会被频繁修改 (每帧更新)
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);

    // ------------------------------------------------------------------
    // 2. 模块初始化
    // ------------------------------------------------------------------
    
    // [场景模块]: 创建 Cornell Box 场景数据
    // 工厂模式：scene.cpp 负责生产数据，main 只负责拿到 Scene 对象
    Scene scene = create_cornell_box();
    
    // [渲染模块]: 将场景数据上传到 GPU 常量内存
    // 这一步之后，GPU 就知道场景里有什么物体了
    init_scene_data(scene.objects, scene.texture_files);

    // [相机模块]: 初始化第一人称相机
    // 参数: 初始位置 (50, 52, 295.6), 初始朝向 (0, -0.04, -1)
    CameraController cam({50, 52, 295.6}, {0, -0.042612, -1});

    // ------------------------------------------------------------------
    // 3. 内存与显存分配 (双缓冲架构的核心)
    // ------------------------------------------------------------------
    size_t size_bytes = w * h * sizeof(Vec);
    
    // [显存 - Buffer A]: 累加缓冲区 (Accumulation Buffer)
    // GPU 正在计算的帧，每一帧的结果都会加到这里面，用于降噪。
    Vec *d_accum;
    cudaMalloc(&d_accum, size_bytes);
    cudaMemset(d_accum, 0, size_bytes); // 初始化为全黑
    
    // [显存 - Buffer B]: 暂存/快照缓冲区 (Staging Buffer)
    // 当 Buffer A 算好一帧后，瞬间拷贝到 Buffer B。
    // CPU 后台线程只从 Buffer B 读取数据，这样 GPU 就可以立刻回去继续写 Buffer A，互不干扰。
    Vec *d_staging;
    cudaMalloc(&d_staging, size_bytes);
    
    // [内存 - Host]: 页锁定内存 (Pinned Memory)
    // 普通 malloc 的内存是可被操作系统交换(swap)的，GPU 访问慢。
    // cudaHostAlloc 申请的内存物理地址固定，PCIe 传输速度最快 (DMA)。
    Vec* h_accum;
    cudaHostAlloc(&h_accum, size_bytes, cudaHostAllocDefault);
    
    // [内存 - Display]: SDL 显示缓冲区 (32位整数格式)
    // 存放最终转换好的 ARGB 像素数据
    uint32_t* pixel_buffer = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    // ------------------------------------------------------------------
    // 4. 启动流水线
    // ------------------------------------------------------------------
    // 初始化 Pipeline 对象，这会启动一个后台 Worker 线程，处于待命状态
    Pipeline pipe;
    pipeline_init(&pipe, h_accum, d_staging, pixel_buffer, w, h);

    // ------------------------------------------------------------------
    // 5. 游戏主循环 (Game Loop)
    // ------------------------------------------------------------------
    SDL_Event e;
    int gpu_frame = 1; // 当前 GPU 正在累积第几帧
    bool quit = false;

    while (!quit) {
        // [新增] 检查 Ctrl+C
        if (g_signal_received) quit = true;

        // --- 1. 事件处理 (包含按键截图) ---
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            
            // [新增] 监听键盘按下事件
            if (e.type == SDL_KEYDOWN) {
                // 如果按下 'P' 键
                // 按 ESC 退出
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    quit = true;
                    printf("\n[System] Saving snapshot...");
                    // 保存当前内存里最新的那帧数据 (h_accum)
                    // 注意：h_accum 是 Worker 上一次搬运过来的，可能比 gpu_frame 滞后一点点，但这不影响截图
                    save_snapshot(h_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture());
                }
            }
        }
        
        // 更新相机位置 (假设每帧 delta_time 为 1.0)
        // update() 内部会检测键盘 WASD 状态
        bool camera_moved = cam.update(1.0f);

        // [关键逻辑]: 累积重置 (Accumulation Reset)
        // 如果相机动了，之前的累积结果就作废了(画面变了)，必须清空重算。
        // 如果相机没动，就继续在旧画面上叠加，让噪点越来越少。
        if (camera_moved) {
            gpu_frame = 1;
            cudaMemset(d_accum, 0, size_bytes); // 异步清零，速度极快
        }

        // --- 步骤 B: GPU 渲染 (Render Dispatch) ---
        // 获取当前帧的相机参数 (View Matrix / FOV)
        CameraParams cam_params = cam.get_params(w, h);
        
        // 发射 CUDA 内核！
        // 这是一个异步调用，CPU 发完指令立刻往下走，不等待 GPU。
        launch_render_kernel(d_accum, w, h, gpu_frame, 16, 16, cam_params);
        
        // --- 步骤 C: 流水线调度 (Pipeline Schedule) ---
        
        // 1. [显存快照]
        // 利用 GPU 内部极高的显存带宽 (300GB/s+)，把 d_accum 瞬间备份到 d_staging。
        // 目的：让后台线程读 d_staging，主线程继续写 d_accum，读写分离。
        cudaMemcpy(d_staging, d_accum, size_bytes, cudaMemcpyDeviceToDevice);
        
        // 2. [同步点]
        // 必须确保快照完成了，d_staging 里的数据是完整的，才能通知后台线程去读。
        cudaDeviceSynchronize();
        
        // 3. [派发任务]
        // 尝试唤醒后台 Worker 线程去处理这一帧的数据 (Copy D2H + Format Convert)。
        // 如果 Worker 还在处理上一帧(忙碌)，这里会直接返回 false，跳过这一帧的显示更新。
        // 这叫 "Drop Frame" (丢帧)，保证了渲染引擎永远不会卡顿。
        pipeline_try_dispatch(&pipe, gpu_frame);

        // --- 步骤 D: 显示上屏 (Display) ---
        
        // 检查 Worker 有没有产出新的成品图片
        if (pipeline_check_frame_ready(&pipe)) {
            // 有新图！上传给 SDL 纹理
            SDL_UpdateTexture(texture, NULL, pixel_buffer, w * sizeof(uint32_t));
            
            // 渲染三部曲：清屏 -> 复制纹理 -> 交换缓冲区(显示)
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
            
            // 更新窗口标题，显示状态
            if (gpu_frame % 10 == 0) {
                char title[128];
                sprintf(title, "CUDA Engine - Frame: %d %s", 
                        gpu_frame, camera_moved ? "(Moving)" : "(Accumulating)");
                SDL_SetWindowTitle(window, title);
            }
        }

        // 帧数计数器自增
        gpu_frame++;
    }

    // ------------------------------------------------------------------
    // 6. 资源清理 (Cleanup)
    // ------------------------------------------------------------------
    // 销毁流水线 (会等待后台线程安全退出)
    pipeline_destroy(&pipe);

    // 释放显存
    cudaFree(d_accum);
    cudaFree(d_staging);
    
    // 释放 Pinned Memory (必须用 cudaFreeHost)
    cudaFreeHost(h_accum);
    
    // 释放普通内存
    free(pixel_buffer);

    // 销毁 SDL 资源
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
