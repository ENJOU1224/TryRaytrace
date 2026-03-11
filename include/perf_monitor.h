#pragma once

#include "renderer.h"

#include <chrono>
#include <cstdio>
#include <string>

/**
 * @file perf_monitor.h
 * @brief 性能监控与日志模块
 *
 * 职责：
 * 1. 记录启动阶段耗时（SDL、场景、BVH、上传、NPU 初始化）。
 * 2. 按帧记录 render/post/present/denoise 等阶段开销。
 * 3. 从 sysfs 读取 GPU/NPU 的近似活跃度，辅助判断瓶颈在“算得慢”还是“没喂满”。
 */

using Clock = std::chrono::steady_clock;

struct StartupPerf {
    // 这些字段记录“程序刚启动时，各阶段一共花了多久”。
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
    // GT0 / GT1 是 Intel GPU 上两个主要图形执行单元的近似指标。
    double gt0_act_freq_mhz = -1.0;
    double gt1_act_freq_mhz = -1.0;
    double gt0_idle_delta_ms = -1.0;
    double gt1_idle_delta_ms = -1.0;
    double gt0_busy_pct = -1.0;
    double gt1_busy_pct = -1.0;
    // NPU 指标同理，读不到时保持 -1。
    double npu_busy_pct = -1.0;
    double npu_freq_mhz = -1.0;
    double npu_mem_bytes = -1.0;
};

std::string make_timestamp();
double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end);
inline float milli_to_float(uint32_t milli_value) {
    return static_cast<float>(milli_value) / 1000.0f;
}

class DeviceMetricsSampler {
public:
    DeviceMetricsSampler();
    DeviceMetrics sample(double frame_ms);

private:
    static double read_double(const char* path, double fallback);
    static double compute_delta(double current, double previous);
    static double estimate_busy_pct(double frame_ms, double idle_delta_ms);

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

class PerfLogger {
public:
    // 构造时就打开日志文件并写表头，后续每帧只负责追加一行。
    explicit PerfLogger(const StartupPerf& startup);
    ~PerfLogger();

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
                   const RenderStats* render_stats);

private:
    FILE* file_ = nullptr;
    std::string path_;
    int pending_rows_ = 0;
};
