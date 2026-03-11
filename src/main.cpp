#include <csignal>
#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>

#include "common.h"     
#include "scene.h"      
#include "camera.h"     
#include "renderer.h"   
#include "input.h"      
#include "image_io.h"   
#include "bvh.h"        
#include "denoiser.h"

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

namespace {

// 默认运行策略：
// 1. 基础性能优先，默认关闭 NPU 降噪。
// 2. 诊断统计默认关闭，避免终端刷屏，也避免内核里频繁原子操作带来的额外开销。
constexpr bool kEnableNpuDenoiser = false;
constexpr bool kEnableDiagnosticStats = false;
constexpr int kProgressPrintInterval = 5;

float milli_to_float(uint32_t milli_value) {
    return static_cast<float>(milli_value) / 1000.0f;
}

/**
 * 将路径追踪累积缓冲区转换为“线性 RGB 帧”。
 *
 * 注意这里故意不做 Gamma 校正:
 * 1. 大多数图像降噪模型更适合吃线性空间数据；
 * 2. Gamma 校正属于显示阶段，不应混入推理输入。
 */
void build_linear_rgb_frame(const Vec* accum, int pixel_count, int frame_index, std::vector<float>& rgb_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        Vec color = accum[i] * (1.0f / frame_index);
        rgb_buffer[i * 3 + 0] = std::max(0.0f, color.x);
        rgb_buffer[i * 3 + 1] = std::max(0.0f, color.y);
        rgb_buffer[i * 3 + 2] = std::max(0.0f, color.z);
    }
}

/**
 * 将线性 RGB 图像转换成 SDL 纹理可直接显示的 ARGB8888。
 * 这里才执行 Tone Mapping + Gamma，保证“推理输入”和“显示输出”职责分离。
 */
void linear_rgb_to_argb8888(const float* rgb_buffer, int pixel_count, uint32_t* pixel_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        const float r = encode_display_value(rgb_buffer[i * 3 + 0]);
        const float g = encode_display_value(rgb_buffer[i * 3 + 1]);
        const float b = encode_display_value(rgb_buffer[i * 3 + 2]);
        pixel_buffer[i] = (255 << 24) |
                          (static_cast<uint32_t>(r * 255.0f + 0.5f) << 16) |
                          (static_cast<uint32_t>(g * 255.0f + 0.5f) << 8) |
                          static_cast<uint32_t>(b * 255.0f + 0.5f);
    }
}

/**
 * 纯渲染模式下的快速显示路径。
 *
 * 当 NPU 降噪关闭时，没有必要先把整帧写入线性 RGB 缓冲区，再做第二次遍历转成 ARGB。
 * 这里直接从累积缓冲区完成“平均化 -> Tone Mapping -> Gamma -> 打包显示”，
 * 可以少一次完整的 CPU 内存遍历和两块大缓冲区的常驻占用。
 */
void accum_to_argb8888(const Vec* accum, int pixel_count, int frame_index, uint32_t* pixel_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        Vec color = accum[i] * (1.0f / frame_index);
        const float r = encode_display_value(color.x);
        const float g = encode_display_value(color.y);
        const float b = encode_display_value(color.z);
        pixel_buffer[i] = (255u << 24) |
                          (static_cast<uint32_t>(r * 255.0f + 0.5f) << 16) |
                          (static_cast<uint32_t>(g * 255.0f + 0.5f) << 8) |
                          static_cast<uint32_t>(b * 255.0f + 0.5f);
    }
}

/**
 * 将原图与降噪结果做渐进混合。
 *
 * 设计原因:
 * 1. 低采样帧数时，降噪结果不一定稳定，直接全量替换很容易“首帧劣化”。
 * 2. 通过 alpha 逐步抬高，可以让用户先看到可信的原图，再慢慢引入 NPU 平滑结果。
 */
void blend_linear_rgb(const float* original_rgb,
                      const float* denoised_rgb,
                      float alpha,
                      int pixel_count,
                      std::vector<float>& blended_rgb) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count * 3; ++i) {
        blended_rgb[i] = original_rgb[i] * (1.0f - alpha) + denoised_rgb[i] * alpha;
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

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
    // 主线程与渲染内核共用同一个队列，避免 USM 分配和内核执行分属不同上下文。
    sycl::queue& q = get_renderer_queue();

    // 分配 USM Shared Memory：CPU 与 GPU 物理共享，零拷贝访问
    Vec* d_accum = sycl::malloc_shared<Vec>(w * h, q);
    std::memset(d_accum, 0, w * h * sizeof(Vec));
    RenderStats* render_stats = nullptr;
    if (kEnableDiagnosticStats) {
        render_stats = sycl::malloc_shared<RenderStats>(1, q);
        *render_stats = {};
    }
    
    // CPU 侧像素映射缓冲区 (用于 SDL 显示)
    std::vector<uint32_t> pixel_buffer(w * h, 0);
    std::vector<float> linear_rgb_frame;
    std::vector<float> blended_rgb_frame;
    if (kEnableNpuDenoiser) {
        linear_rgb_frame.resize(static_cast<size_t>(w) * h * 3, 0.0f);
        blended_rgb_frame.resize(static_cast<size_t>(w) * h * 3, 0.0f);
    }

    FrameDenoiser denoiser;
    std::string last_denoiser_status;
    if (kEnableNpuDenoiser) {
        denoiser.initialize(w, h);
        last_denoiser_status = denoiser.status();
    } else {
        last_denoiser_status = "[系统] 当前已关闭 NPU 降噪。";
    }
    if (!last_denoiser_status.empty()) {
        std::cout << last_denoiser_status << std::endl;
    }

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
            if (kEnableNpuDenoiser) {
                denoiser.reset();
            }
        }

        // [B] 发射渲染内核
        CameraParams cam_params = cam.get_params(w, h);
        // 建议 tx=16, ty=8 以匹配 Intel Xe2 硬件 Sub-group 布局
        if (render_stats) {
            *render_stats = {};
        }
        launch_render_kernel(d_accum, w, h, gpu_frame, 16, 8, cam_params, render_stats);
        
        // [C] 图像后处理与 NPU 降噪
        if (kEnableNpuDenoiser) {
            build_linear_rgb_frame(d_accum, w * h, gpu_frame, linear_rgb_frame);

            if (denoiser.should_run(gpu_frame) &&
                !denoiser.run(linear_rgb_frame.data(), w, h)) {
                if (denoiser.status() != last_denoiser_status) {
                    std::cout << denoiser.status() << std::endl;
                    last_denoiser_status = denoiser.status();
                }
            }

            const float* active_rgb = linear_rgb_frame.data();
            if (denoiser.has_output()) {
                const float alpha = denoiser.blend_alpha(gpu_frame);
                if (alpha > 0.0f) {
                    blend_linear_rgb(linear_rgb_frame.data(),
                                     denoiser.output_rgb().data(),
                                     alpha,
                                     w * h,
                                     blended_rgb_frame);
                    active_rgb = blended_rgb_frame.data();
                }
            }
            linear_rgb_to_argb8888(active_rgb, w * h, pixel_buffer.data());
        } else {
            accum_to_argb8888(d_accum, w * h, gpu_frame, pixel_buffer.data());
        }

        // [D] 更新 SDL 屏幕
        SDL_UpdateTexture(texture, NULL, pixel_buffer.data(), w * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        
        // [E] 性能统计与反馈
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = current_time - last_time;
        last_time = current_time;
        fps = 0.9f * fps + 0.1f * (1.0f / delta.count()); 

        if (gpu_frame % kProgressPrintInterval == 0) {
            char title[256];
            sprintf(title, "[%s] FPS: %.1f | Frame: %d | Denoise: %s",
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    fps, gpu_frame,
                    (kEnableNpuDenoiser && denoiser.is_enabled()) ? denoiser.device_name().c_str() : "OFF");
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
                printf("\r>> [SYCL] Frame %d | FPS: %.1f", gpu_frame, fps);
                fflush(stdout);
            }
        }
        gpu_frame++;
    }

    // 退出前自动存档
    save_snapshot(d_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture()); 

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
