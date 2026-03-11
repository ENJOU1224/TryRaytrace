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
// 1. 默认启用异步 NPU 降噪流水线。
// 2. 诊断统计默认关闭，避免终端刷屏，也避免内核里频繁原子操作带来的额外开销。
constexpr bool kEnableNpuDenoiser = true;
constexpr bool kEnableDiagnosticStats = false;
constexpr int kProgressPrintInterval = 5;
constexpr int kStaticPresentInterval = 2;
constexpr int kPerfFlushInterval = 30;
constexpr int kRenderTileWidth = 8;
constexpr int kRenderTileHeight = 8;

using Clock = std::chrono::steady_clock;

struct StartupPerf {
    double sdl_setup_ms = 0.0;
    double scene_create_ms = 0.0;
    double bvh_build_ms = 0.0;
    double upload_ms = 0.0;
    double denoiser_init_ms = 0.0;
    int display_width = 0;
    int display_height = 0;
    int render_width = 0;
    int render_height = 0;
};

struct DeviceMetrics {
    double gt0_act_freq_mhz = -1.0;
    double gt1_act_freq_mhz = -1.0;
    double gt0_idle_delta_ms = -1.0;
    double gt1_idle_delta_ms = -1.0;
    double gt0_busy_pct = -1.0;
    double gt1_busy_pct = -1.0;
    double npu_busy_pct = -1.0;
    double npu_freq_mhz = -1.0;
    double npu_mem_bytes = -1.0;
};

class DeviceMetricsSampler {
public:
    DeviceMetricsSampler()
        : gt0_act_freq_path_("/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/act_freq"),
          gt1_act_freq_path_("/sys/devices/pci0000:00/0000:00:02.0/tile0/gt1/freq0/act_freq"),
          gt0_idle_residency_path_("/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/gtidle/idle_residency_ms"),
          gt1_idle_residency_path_("/sys/devices/pci0000:00/0000:00:02.0/tile0/gt1/gtidle/idle_residency_ms"),
          npu_busy_time_path_("/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us"),
          npu_freq_path_("/sys/devices/pci0000:00/0000:00:0b.0/npu_current_frequency_mhz"),
          npu_mem_path_("/sys/devices/pci0000:00/0000:00:0b.0/npu_memory_utilization") {
        previous_gt0_idle_ms_ = read_double(gt0_idle_residency_path_, -1.0);
        previous_gt1_idle_ms_ = read_double(gt1_idle_residency_path_, -1.0);
        previous_npu_busy_us_ = read_double(npu_busy_time_path_, -1.0);
    }

    DeviceMetrics sample(double frame_ms) {
        DeviceMetrics metrics;
        metrics.gt0_act_freq_mhz = read_double(gt0_act_freq_path_, -1.0);
        metrics.gt1_act_freq_mhz = read_double(gt1_act_freq_path_, -1.0);
        metrics.npu_freq_mhz = read_double(npu_freq_path_, -1.0);
        metrics.npu_mem_bytes = read_double(npu_mem_path_, -1.0);

        const double current_gt0_idle_ms = read_double(gt0_idle_residency_path_, -1.0);
        const double current_gt1_idle_ms = read_double(gt1_idle_residency_path_, -1.0);
        const double current_npu_busy_us = read_double(npu_busy_time_path_, -1.0);

        metrics.gt0_idle_delta_ms = compute_delta(current_gt0_idle_ms, previous_gt0_idle_ms_);
        metrics.gt1_idle_delta_ms = compute_delta(current_gt1_idle_ms, previous_gt1_idle_ms_);
        metrics.gt0_busy_pct = estimate_busy_pct(frame_ms, metrics.gt0_idle_delta_ms);
        metrics.gt1_busy_pct = estimate_busy_pct(frame_ms, metrics.gt1_idle_delta_ms);

        const double npu_busy_delta_us = compute_delta(current_npu_busy_us, previous_npu_busy_us_);
        if (npu_busy_delta_us >= 0.0 && frame_ms > 0.0) {
            metrics.npu_busy_pct = std::clamp((npu_busy_delta_us / (frame_ms * 1000.0)) * 100.0, 0.0, 100.0);
        }

        previous_gt0_idle_ms_ = current_gt0_idle_ms;
        previous_gt1_idle_ms_ = current_gt1_idle_ms;
        previous_npu_busy_us_ = current_npu_busy_us;
        return metrics;
    }

private:
    static double read_double(const char* path, double fallback) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return fallback;
        }

        double value = fallback;
        file >> value;
        return file.fail() ? fallback : value;
    }

    static double compute_delta(double current, double previous) {
        if (current < 0.0 || previous < 0.0) {
            return -1.0;
        }
        double delta = current - previous;
        return delta >= 0.0 ? delta : -1.0;
    }

    static double estimate_busy_pct(double frame_ms, double idle_delta_ms) {
        if (frame_ms <= 0.0 || idle_delta_ms < 0.0) {
            return -1.0;
        }
        return std::clamp((1.0 - idle_delta_ms / frame_ms) * 100.0, 0.0, 100.0);
    }

    const char* gt0_act_freq_path_;
    const char* gt1_act_freq_path_;
    const char* gt0_idle_residency_path_;
    const char* gt1_idle_residency_path_;
    const char* npu_busy_time_path_;
    const char* npu_freq_path_;
    const char* npu_mem_path_;

    double previous_gt0_idle_ms_ = -1.0;
    double previous_gt1_idle_ms_ = -1.0;
    double previous_npu_busy_us_ = -1.0;
};

std::string make_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* tm = std::localtime(&now);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", tm);
    return buffer;
}

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/**
 * 运行时性能日志。
 *
 * 目标不是做复杂 profiling，而是稳定记录主循环各阶段开销。
 * 这样用户跑完之后，我们可以直接看：
 * 1. GPU kernel 是否是绝对瓶颈；
 * 2. CPU 后处理 / SDL 提交是否吞掉了太多时间；
 * 3. 在开启降噪时，NPU 推理到底占了多少比例。
 */
class PerfLogger {
public:
    explicit PerfLogger(const StartupPerf& startup) {
        std::error_code ec;
        std::filesystem::create_directories("logs", ec);
        if (ec) {
            std::cerr << "[Perf] Failed to create logs directory: " << ec.message() << std::endl;
            return;
        }

        path_ = "logs/perf_" + make_timestamp() + ".csv";
        file_ = std::fopen(path_.c_str(), "w");
        if (!file_) {
            std::cerr << "[Perf] Failed to open log file: " << path_ << std::endl;
            return;
        }

        std::fprintf(file_, "# sdl_setup_ms=%.3f\n", startup.sdl_setup_ms);
        std::fprintf(file_, "# scene_create_ms=%.3f\n", startup.scene_create_ms);
        std::fprintf(file_, "# bvh_build_ms=%.3f\n", startup.bvh_build_ms);
        std::fprintf(file_, "# upload_ms=%.3f\n", startup.upload_ms);
        std::fprintf(file_, "# denoiser_init_ms=%.3f\n", startup.denoiser_init_ms);
        std::fprintf(file_, "# display_resolution=%dx%d\n", startup.display_width, startup.display_height);
        std::fprintf(file_, "# render_resolution=%dx%d\n", startup.render_width, startup.render_height);
        std::fprintf(file_,
                     "frame,presented,camera_moved,save_request,input_ms,render_ms,post_ms,present_ms,denoise_ms,total_ms,fps,"
                     "gt0_act_freq_mhz,gt1_act_freq_mhz,gt0_idle_delta_ms,gt1_idle_delta_ms,gt0_busy_pct,gt1_busy_pct,"
                     "npu_busy_pct,npu_freq_mhz,npu_mem_bytes,"
                     "dl_clamp,eh_clamp,eh_primary_clamp,eh_indirect_clamp,tp_diffuse,tp_specular,tp_refract,tp_rr,final_firefly_clamp,nan_or_inf\n");
        std::fflush(file_);

        std::cout << "[Perf] Logging frame timings to " << path_ << std::endl;
    }

    ~PerfLogger() {
        if (file_) {
            std::fflush(file_);
            std::fclose(file_);
        }
    }

    void log_frame(int frame_index,
                   bool presented,
                   bool camera_moved,
                   bool save_request,
                   double input_ms,
                   double render_ms,
                   double post_ms,
                   double present_ms,
                   double denoise_ms,
                   double total_ms,
                   double fps,
                   const DeviceMetrics& device_metrics,
                   const RenderStats* render_stats) {
        if (!file_) {
            return;
        }

        auto stat_or_neg1 = [render_stats](uint32_t RenderStats::*member) -> int {
            if (!render_stats) {
                return -1;
            }
            return static_cast<int>(render_stats->*member);
        };

        std::fprintf(file_,
                     "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f,%.0f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                     frame_index,
                     presented ? 1 : 0,
                     camera_moved ? 1 : 0,
                     save_request ? 1 : 0,
                     input_ms,
                     render_ms,
                     post_ms,
                     present_ms,
                     denoise_ms,
                     total_ms,
                     fps,
                     device_metrics.gt0_act_freq_mhz,
                     device_metrics.gt1_act_freq_mhz,
                     device_metrics.gt0_idle_delta_ms,
                     device_metrics.gt1_idle_delta_ms,
                     device_metrics.gt0_busy_pct,
                     device_metrics.gt1_busy_pct,
                     device_metrics.npu_busy_pct,
                     device_metrics.npu_freq_mhz,
                     device_metrics.npu_mem_bytes,
                     stat_or_neg1(&RenderStats::direct_light_clamp_count),
                     stat_or_neg1(&RenderStats::emissive_hit_clamp_count),
                     stat_or_neg1(&RenderStats::emissive_hit_primary_clamp_count),
                     stat_or_neg1(&RenderStats::emissive_hit_indirect_clamp_count),
                     stat_or_neg1(&RenderStats::throughput_clamp_diffuse_count),
                     stat_or_neg1(&RenderStats::throughput_clamp_specular_count),
                     stat_or_neg1(&RenderStats::throughput_clamp_refract_count),
                     stat_or_neg1(&RenderStats::throughput_clamp_rr_count),
                     stat_or_neg1(&RenderStats::final_firefly_clamp_count),
                     stat_or_neg1(&RenderStats::nan_or_inf_pixels));

        ++pending_rows_;
        if (pending_rows_ >= kPerfFlushInterval) {
            std::fflush(file_);
            pending_rows_ = 0;
        }
    }

private:
    FILE* file_ = nullptr;
    std::string path_;
    int pending_rows_ = 0;
};

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

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // 注册信号处理，支持 Ctrl+C 安全退出
    std::signal(SIGINT, signal_handler);

    StartupPerf startup_perf;

    // 1. 系统参数配置
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
    // 创建场景数据 (Cornell Box + OBJ Models)
    auto scene_begin = Clock::now();
    Scene scene = create_cornell_box();
    startup_perf.scene_create_ms = elapsed_ms(scene_begin, Clock::now());
    
    // 构建 BVH 加速结构 (线性化处理)
    auto bvh_begin = Clock::now();
    BVH bvh; 
    bvh.build(scene.objects);
    startup_perf.bvh_build_ms = elapsed_ms(bvh_begin, Clock::now());

    // 筛选光源索引 (用于 NEE 显式采样优化)
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

    // 分配 USM Shared Memory：CPU 与 GPU 物理共享，零拷贝访问
    Vec* d_accum = sycl::malloc_shared<Vec>(render_w * render_h, q);
    std::memset(d_accum, 0, static_cast<size_t>(render_w) * render_h * sizeof(Vec));
    RenderStats* render_stats = nullptr;
    if (kEnableDiagnosticStats) {
        render_stats = sycl::malloc_shared<RenderStats>(1, q);
        *render_stats = {};
    }
    
    // CPU 侧像素映射缓冲区 (用于 SDL 显示)
    std::vector<uint32_t> pixel_buffer(render_w * render_h, 0);
    std::vector<float> linear_rgb_frame;
    if (kEnableNpuDenoiser) {
        linear_rgb_frame.resize(static_cast<size_t>(render_w) * render_h * 3, 0.0f);
    }

    AsyncFrameDenoiser denoiser;
    std::string last_denoiser_status;
    auto denoiser_begin = Clock::now();
    if (kEnableNpuDenoiser) {
        denoiser.initialize(render_w, render_h);
        last_denoiser_status = denoiser.status();
    } else {
        last_denoiser_status = "[系统] 当前已关闭 NPU 降噪。";
    }
    if (!last_denoiser_status.empty()) {
        std::cout << last_denoiser_status << std::endl;
    }
    startup_perf.denoiser_init_ms = elapsed_ms(denoiser_begin, Clock::now());

    PerfLogger perf_logger(startup_perf);
    DeviceMetricsSampler device_metrics_sampler;

    InputManager input; 
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
        if (state.save_request) {
            save_snapshot(d_accum, render_w, render_h, gpu_frame, cam.get_focus_dist(), cam.get_aperture());
        }
        if (state.quit) quit = true;

        // 相机移动时重置累加器，确保画面不产生拖影
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
            if (kEnableNpuDenoiser) {
                build_linear_rgb_frame(d_accum, render_w * render_h, gpu_frame, linear_rgb_frame);

                auto denoise_begin_frame = Clock::now();
                denoiser.submit_frame(gpu_frame, linear_rgb_frame.data());
                denoise_ms = elapsed_ms(denoise_begin_frame, Clock::now());

                if (denoiser.status() != last_denoiser_status) {
                    std::cout << denoiser.status() << std::endl;
                    last_denoiser_status = denoiser.status();
                }

                const float* active_rgb = linear_rgb_frame.data();
                if (denoiser.has_result()) {
                    active_rgb = denoiser.latest_result().data();
                }
                linear_rgb_to_argb8888(active_rgb, render_w * render_h, pixel_buffer.data());
            } else {
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
            sprintf(title, "[%s] FPS: %.1f | Frame: %d | Denoise: %s | Lag: %d",
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    fps, gpu_frame,
                    (kEnableNpuDenoiser && denoiser.is_enabled()) ? denoiser.device_name().c_str() : "OFF",
                    (kEnableNpuDenoiser && denoiser.has_result()) ? (gpu_frame - denoiser.latest_result_frame_id()) : 0);
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
                       (kEnableNpuDenoiser && denoiser.has_result()) ? (gpu_frame - denoiser.latest_result_frame_id()) : 0);
                fflush(stdout);
            }
        }
        gpu_frame++;
    }

    // 退出前自动存档
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
