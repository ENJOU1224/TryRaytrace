#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>

#include "common.h"     
#include "scene.h"      
#include "camera.h"     
#include "renderer.h"   
#include "input.h"      
#include "image_io.h"   
#include "bvh.h"        
#include "async_denoiser.h"
#include "frame_processing.h"
#include "perf_monitor.h"

/**
 * @file main.cpp
 * @brief 系统主入口与调度器
 * 
 * 职责:
 * 1. 资源管理: 初始化 SDL2 窗口、SYCL 队列及 USM 共享内存。
 * 2. 场景驱动: 调用各模块完成场景构建、BVH 划分及数据上传。
 * 3. 实时渲染循环: 协调 GPU 计算、CPU 后处理与屏幕显示。
 *
 * 对初学者来说，可以把 main.cpp 当成“总导演”来看：
 * - 它自己不负责具体渲染数学
 * - 它只负责安排“谁先做、谁后做、结果交给谁”
 *
 * 推荐阅读顺序：
 * 1. 看启动阶段准备了哪些资源
 * 2. 看 while 循环每一帧分成哪几个阶段
 * 3. 最后看退出时如何清理资源
 */

std::atomic<bool> quit(false);
// Ctrl+C 时不要立刻粗暴中断，而是只把 quit 设为 true，
// 让主循环自然收尾、保存快照、释放资源。
void signal_handler(int signal) { if (signal == SIGINT) quit = true; }

namespace {

// 默认运行策略：
// 1. 默认启用异步 NPU 降噪流水线。
// 2. 诊断统计默认关闭，避免终端刷屏，也避免内核里频繁原子操作带来的额外开销。
constexpr bool kDefaultEnableNpuDenoiser = true;
constexpr bool kEnableDiagnosticStats = false;
// 终端和标题栏不需要每帧都刷新，否则输出噪声太大。
constexpr int kProgressPrintInterval = 5;
// 静止观察时，隔几帧再真正提交一次屏幕显示，减少 CPU 和 SDL 开销。
constexpr int kStaticPresentInterval = 2;
// 一个 work-group 覆盖多少像素。
// 实际最佳值和硬件有关，这里是当前机器上验证过的经验值。
constexpr int kRenderTileWidth = 8;
constexpr int kRenderTileHeight = 8;

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }

    std::string text(value);
    for (char& ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }

    return text == "1" || text == "true" || text == "yes" || text == "on";
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // 注册信号处理，支持 Ctrl+C 安全退出
    std::signal(SIGINT, signal_handler);

    const bool enable_npu_denoiser =
        kDefaultEnableNpuDenoiser && !env_flag_enabled("LUNAR_DISABLE_NPU_DENOISER");

    StartupPerf startup_perf;

    // 1. 系统参数配置
    // display_* 是窗口大小。
    // render_* 是实际渲染分辨率。
    // 当前两者相同；以后如果想做“低分辨率渲染再放大显示”，可以把它们拆开。
    const int display_w = 1200;
    const int display_h = 800;
    const int render_w = display_w;
    const int render_h = display_h;
    startup_perf.display_width = display_w;
    startup_perf.display_height = display_h;
    startup_perf.render_width = render_w;
    startup_perf.render_height = render_h;
    
    // 初始化 SDL2 视频子系统
    auto sdl_begin = Clock::now();
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[SDL Error] Initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    // 创建图形窗口与渲染上下文
    SDL_Window* window = SDL_CreateWindow("Lunar-Raytrace (oneAPI/SYCL)", 
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                          display_w, display_h, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, 
                                             SDL_TEXTUREACCESS_STREAMING, render_w, render_h);
    startup_perf.sdl_setup_ms = elapsed_ms(sdl_begin, Clock::now());

    // 2. 场景与渲染引擎初始化
    // 到这里开始，程序才真正进入“渲染器的准备阶段”。
    // 顺序是：
    // Scene -> BVH -> Light List -> 上传到 GPU/USM
    auto scene_begin = Clock::now();
    Scene scene = create_cornell_box();
    startup_perf.scene_create_ms = elapsed_ms(scene_begin, Clock::now());
    
    // 构建 BVH 加速结构 (线性化处理)
    auto bvh_begin = Clock::now();
    BVH bvh; 
    bvh.build(scene.objects);
    startup_perf.bvh_build_ms = elapsed_ms(bvh_begin, Clock::now());

    // 筛选光源索引 (用于 NEE 显式采样优化)
    // 之所以单独存索引，而不是每次渲染临时遍历所有物体，
    // 是为了让 GPU 端能更快地“只在灯的集合里随机选一个”。
    std::vector<int> light_indices;
    for (size_t i = 0; i < scene.objects.size(); i++) {
        if (scene.objects[i].emission.norm_len() > 0.1f) {
            light_indices.push_back((int)i);
        }
    }

    // 初始化 SYCL 渲染核心并上传数据到 USM (统一内存)
    auto upload_begin = Clock::now();
    init_scene_data(scene.objects, scene.texture_files, bvh.get_nodes(), light_indices);
    startup_perf.upload_ms = elapsed_ms(upload_begin, Clock::now());

    // 初始化相机控制器 (第一人称视角)
    CameraController cam({50, 50, 295.6}, {0, 0, -1});

    // 3. 运行时资源准备
    // 主线程与渲染内核共用同一个队列，避免 USM 分配和内核执行分属不同上下文。
    sycl::queue& q = get_renderer_queue();

    // 分配 USM Shared Memory：CPU 与 GPU 物理共享，零拷贝访问。
    // d_accum 是最核心的帧缓冲：
    // - GPU 每渲染完一帧，就把这一帧颜色“累加”进去
    // - CPU 显示时再除以 gpu_frame，得到当前平均结果
    Vec* d_accum = sycl::malloc_shared<Vec>(render_w * render_h, q);
    std::memset(d_accum, 0, static_cast<size_t>(render_w) * render_h * sizeof(Vec));
    RenderStats* render_stats = nullptr;
    if (kEnableDiagnosticStats) {
        render_stats = sycl::malloc_shared<RenderStats>(1, q);
        *render_stats = {};
    }
    
    // CPU 侧像素映射缓冲区 (用于 SDL 显示)
    // SDL 纹理吃的是 ARGB8888 整数像素，不是 Vec / float，所以需要一个中间显示缓冲。
    std::vector<uint32_t> pixel_buffer(render_w * render_h, 0);
    std::vector<float> linear_rgb_frame;
    if (enable_npu_denoiser) {
        // 降噪器更适合接收“线性 RGB 浮点图像”，所以这里额外准备一个 3 通道 float 缓冲。
        linear_rgb_frame.resize(static_cast<size_t>(render_w) * render_h * 3, 0.0f);
    }

    AsyncFrameDenoiser denoiser;
    std::string last_denoiser_status;
    auto denoiser_begin = Clock::now();
    if (enable_npu_denoiser) {
        denoiser.initialize(render_w, render_h);
        last_denoiser_status = denoiser.status();
    } else {
        last_denoiser_status = "[系统] 当前已关闭 NPU 降噪（可通过环境变量临时切换）。";
    }
    if (!last_denoiser_status.empty()) {
        std::cout << last_denoiser_status << std::endl;
    }
    startup_perf.denoiser_init_ms = elapsed_ms(denoiser_begin, Clock::now());

    PerfLogger perf_logger(startup_perf);
    DeviceMetricsSampler device_metrics_sampler;

    InputManager input; 
    // gpu_frame 不是“显示了多少帧”，而是“当前这张图已经累积了多少轮采样”。
    int gpu_frame = 1;
    
    // FPS 与性能统计
    auto last_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;
    bool frame_presented = false;

    // ------------------------------------------------------------------
    // 4. 实时渲染主循环
    // ------------------------------------------------------------------
    while (!quit) {
        auto frame_begin = Clock::now();

        // [A] 处理用户输入
        auto input_begin = Clock::now();
        InputState state = input.process_events(cam);
        auto input_end = Clock::now();
        
        // 保存当前快照
        // 注意这里保存的是“当前累积结果”，因此图像质量取决于 gpu_frame 已累积了多少。
        if (state.save_request) {
            save_snapshot(d_accum, render_w, render_h, gpu_frame, cam.get_focus_dist(), cam.get_aperture());
        }
        if (state.quit) quit = true;

        // 相机移动时重置累加器，确保画面不产生拖影。
        // 原因很重要：
        // 旧视角采样到的颜色和新视角已经不是同一个问题了，不能继续平均。
        if (state.camera_moved) {
            gpu_frame = 1;
            std::memset(d_accum, 0, static_cast<size_t>(render_w) * render_h * sizeof(Vec));
        }

        // [B] 发射渲染内核
        CameraParams cam_params = cam.get_params(render_w, render_h);
        // 建议 tx=16, ty=8 以匹配 Intel Xe2 硬件 Sub-group 布局
        if (render_stats) {
            *render_stats = {};
        }
        auto render_begin = Clock::now();
        launch_render_kernel(d_accum, render_w, render_h, gpu_frame, kRenderTileWidth, kRenderTileHeight, cam_params, render_stats);
        auto render_end = Clock::now();
        
        // [C] 图像后处理与屏幕提交
        // 渲染帧数增长时，没有必要每一帧都做一次整屏 CPU 转换和 SDL 提交。
        // 静止观察时隔帧显示，能显著减少主循环的 CPU 开销，同时不影响采样累积。
        const bool force_present = state.camera_moved || (gpu_frame <= 2);
        const bool should_present = force_present || (gpu_frame % kStaticPresentInterval == 0);
        frame_presented = false;
        double post_ms = 0.0;
        double present_ms = 0.0;
        double denoise_ms = 0.0;

        if (should_present) {
            auto post_begin = Clock::now();
            if (enable_npu_denoiser) {
                // 1. 把累积缓冲转成“平均后的线性 RGB”
                build_linear_rgb_frame(d_accum, render_w * render_h, gpu_frame, linear_rgb_frame);

                auto denoise_begin_frame = Clock::now();
                // 2. 把最新线性帧提交给后台 NPU 线程。
                //    这里不等待推理完成，所以主线程不会被卡住。
                denoiser.submit_frame(gpu_frame, linear_rgb_frame.data());
                denoise_ms = elapsed_ms(denoise_begin_frame, Clock::now());

                if (denoiser.status() != last_denoiser_status) {
                    std::cout << denoiser.status() << std::endl;
                    last_denoiser_status = denoiser.status();
                }

                // 3. 显示时优先使用“最近已经完成的降噪结果”。
                //    如果后台还没出结果，就先显示当前原图。
                const float* active_rgb = linear_rgb_frame.data();
                if (denoiser.has_result()) {
                    active_rgb = denoiser.latest_result().data();
                }
                linear_rgb_to_argb8888(active_rgb, render_w * render_h, pixel_buffer.data());
            } else {
                // 关闭降噪时直接走最快路径，少一次中间缓冲遍历。
                accum_to_argb8888(d_accum, render_w * render_h, gpu_frame, pixel_buffer.data());
            }
            auto post_end = Clock::now();
            post_ms = elapsed_ms(post_begin, post_end);

            // [D] 更新 SDL 屏幕
            auto present_begin = Clock::now();
            SDL_UpdateTexture(texture, NULL, pixel_buffer.data(), render_w * sizeof(uint32_t));
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
            auto present_end = Clock::now();
            present_ms = elapsed_ms(present_begin, present_end);
            frame_presented = true;
        }
        
        // [E] 性能统计与反馈
        // 这里用的是简单平滑 FPS，而不是“上一帧瞬时 FPS”。
        // 这样数字更稳定，不会每一帧剧烈抖动。
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = current_time - last_time;
        last_time = current_time;
        fps = 0.9f * fps + 0.1f * (1.0f / delta.count()); 
        auto frame_end = Clock::now();
        const double total_frame_ms = elapsed_ms(frame_begin, frame_end);
        const DeviceMetrics device_metrics = device_metrics_sampler.sample(total_frame_ms);

        perf_logger.log_frame(gpu_frame,
                              frame_presented,
                              state.camera_moved,
                              state.save_request,
                              elapsed_ms(input_begin, input_end),
                              elapsed_ms(render_begin, render_end),
                              post_ms,
                              present_ms,
                              denoise_ms,
                              total_frame_ms,
                              fps,
                              device_metrics,
                              render_stats);

        if (gpu_frame % kProgressPrintInterval == 0) {
            char title[256];
            // Lag = 当前正在累积的帧编号 - 最近完成降噪结果的帧编号。
            // 它能直观看出异步 NPU 结果比渲染主线程“落后了几帧”。
            sprintf(title, "[%s] FPS: %.1f | Frame: %d | Denoise: %s | Lag: %d",
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    fps, gpu_frame,
                    (enable_npu_denoiser && denoiser.is_enabled()) ? denoiser.device_name().c_str() : "OFF",
                    (enable_npu_denoiser && denoiser.has_result()) ? (gpu_frame - denoiser.latest_result_frame_id()) : 0);
            SDL_SetWindowTitle(window, title);

            if (kEnableDiagnosticStats && render_stats) {
                printf("\r>> [SYCL] Frame %d | FPS: %.1f | Clamp[DL:%u EH:%u(P:%u I:%u) TPd:%u TPs:%u TPr:%u TPrr:%u F:%u NaN:%u] | Hit[EH P:%u I:%u] | Max[DL:%.2f EH:%.2f(P:%.2f I:%.2f) TP:%.2f C:%.2f]",
                       gpu_frame,
                       fps,
                       render_stats->direct_light_clamp_count,
                       render_stats->emissive_hit_clamp_count,
                       render_stats->emissive_hit_primary_clamp_count,
                       render_stats->emissive_hit_indirect_clamp_count,
                       render_stats->throughput_clamp_diffuse_count,
                       render_stats->throughput_clamp_specular_count,
                       render_stats->throughput_clamp_refract_count,
                       render_stats->throughput_clamp_rr_count,
                       render_stats->final_firefly_clamp_count,
                       render_stats->nan_or_inf_pixels,
                       render_stats->emissive_hit_primary_count,
                       render_stats->emissive_hit_indirect_count,
                       milli_to_float(render_stats->max_direct_light_lum_milli),
                       milli_to_float(render_stats->max_emissive_hit_lum_milli),
                       milli_to_float(render_stats->max_emissive_hit_primary_lum_milli),
                       milli_to_float(render_stats->max_emissive_hit_indirect_lum_milli),
                       milli_to_float(render_stats->max_throughput_component_milli),
                       milli_to_float(render_stats->max_final_color_lum_milli));
                fflush(stdout);
            } else {
                printf("\r>> [SYCL] Frame %d | FPS: %.1f | Present: %s | DenoiseLag: %d",
                       gpu_frame,
                       fps,
                       frame_presented ? "yes" : "skip",
                       (enable_npu_denoiser && denoiser.has_result()) ? (gpu_frame - denoiser.latest_result_frame_id()) : 0);
                fflush(stdout);
            }
        }
        // 这一轮采样已经累进到 d_accum 里，下一轮就把 frame 序号加 1。
        gpu_frame++;
    }

    // 退出前自动存档，防止 Ctrl+C 或窗口关闭后当前结果直接丢失。
    save_snapshot(d_accum, render_w, render_h, gpu_frame, cam.get_focus_dist(), cam.get_aperture());

    // 5. 资源清理
    sycl::free(d_accum, q); 
    if (render_stats) {
        sycl::free(render_stats, q);
    }
    SDL_DestroyTexture(texture); 
    SDL_DestroyRenderer(renderer); 
    SDL_DestroyWindow(window); 
    SDL_Quit();

    std::cout << "\n[System] Render session ended safely." << std::endl;
    return 0;
}
