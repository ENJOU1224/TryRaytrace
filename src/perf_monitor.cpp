#include "perf_monitor.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

// 日志不是每帧都 fflush。
// 原因：
// 1. flush 太频繁会把本来很轻的性能记录变成可见的 IO 开销。
// 2. 30 帧一刷，对当前项目足够安全，也能保证出问题时日志不会丢太多。
constexpr int kPerfFlushInterval = 30;

}

/**
 * 生成统一的时间戳字符串。
 *
 * 整个项目里：
 * - 快照文件
 * - 性能日志
 * 都沿用相同格式，方便人工对照。
 */
std::string make_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* tm = std::localtime(&now);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", tm);
    return buffer;
}

/// 统一的“时间点差值 -> 毫秒”辅助函数。
double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/**
 * 采样器构造时先读一次 sysfs 当前值，作为后续增量计算的基线。
 *
 * 注意这里读取的是“累计计数器”：
 * - GPU idle residency
 * - NPU busy time
 *
 * 真正有意义的是相邻两帧之间的增量，而不是绝对值本身。
 */
DeviceMetricsSampler::DeviceMetricsSampler()
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

DeviceMetrics DeviceMetricsSampler::sample(double frame_ms) {
    DeviceMetrics metrics;

    // 1. 先读“瞬时可读值”：
    //    - 频率
    //    - NPU 内存占用
    metrics.gt0_act_freq_mhz = read_double(gt0_act_freq_path_, -1.0);
    metrics.gt1_act_freq_mhz = read_double(gt1_act_freq_path_, -1.0);
    metrics.npu_freq_mhz = read_double(npu_freq_path_, -1.0);
    metrics.npu_mem_bytes = read_double(npu_mem_path_, -1.0);

    // 2. 再读“累计计数器”：
    //    - GPU idle 累积时间
    //    - NPU busy 累积时间
    const double current_gt0_idle_ms = read_double(gt0_idle_residency_path_, -1.0);
    const double current_gt1_idle_ms = read_double(gt1_idle_residency_path_, -1.0);
    const double current_npu_busy_us = read_double(npu_busy_time_path_, -1.0);

    // 3. 把累计计数器转换成“本帧增量”
    metrics.gt0_idle_delta_ms = compute_delta(current_gt0_idle_ms, previous_gt0_idle_ms_);
    metrics.gt1_idle_delta_ms = compute_delta(current_gt1_idle_ms, previous_gt1_idle_ms_);

    // 4. 再用“本帧总时长 - idle 增量”估算 GPU 忙碌率。
    //    这不是像 intel_gpu_top 那样的官方精确占用率，但足够用于趋势判断。
    metrics.gt0_busy_pct = estimate_busy_pct(frame_ms, metrics.gt0_idle_delta_ms);
    metrics.gt1_busy_pct = estimate_busy_pct(frame_ms, metrics.gt1_idle_delta_ms);

    // 5. NPU 也是同理：用 busy_time 的增量估算本帧忙碌百分比。
    const double npu_busy_delta_us = compute_delta(current_npu_busy_us, previous_npu_busy_us_);
    if (npu_busy_delta_us >= 0.0 && frame_ms > 0.0) {
        metrics.npu_busy_pct = std::clamp((npu_busy_delta_us / (frame_ms * 1000.0)) * 100.0, 0.0, 100.0);
    }

    // 6. 更新上一帧基线，为下次采样准备。
    previous_gt0_idle_ms_ = current_gt0_idle_ms;
    previous_gt1_idle_ms_ = current_gt1_idle_ms;
    previous_npu_busy_us_ = current_npu_busy_us;
    return metrics;
}

/**
 * 从 sysfs 文件中读取一个 double。
 *
 * 失败时返回 fallback，而不是抛异常。
 * 这样主程序在：
 * - 某些节点不存在
 * - 权限不够
 * - 某些驱动版本字段缺失
 * 时也能继续运行，只是对应指标显示为 -1。
 */
double DeviceMetricsSampler::read_double(const char* path, double fallback) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return fallback;
    }

    double value = fallback;
    file >> value;
    return file.fail() ? fallback : value;
}

/// 将“累计值”转换为“本帧增量”。
double DeviceMetricsSampler::compute_delta(double current, double previous) {
    if (current < 0.0 || previous < 0.0) {
        return -1.0;
    }
    double delta = current - previous;
    return delta >= 0.0 ? delta : -1.0;
}

/**
 * 用一帧内 idle 增量估算忙碌率。
 *
 * 公式大致是：
 *   busy = 1 - idle_delta / frame_ms
 *
 * 这不是硬件官方统计，但对于“当前瓶颈是否在 GPU”这个问题已经足够有判断力。
 */
double DeviceMetricsSampler::estimate_busy_pct(double frame_ms, double idle_delta_ms) {
    if (frame_ms <= 0.0 || idle_delta_ms < 0.0) {
        return -1.0;
    }
    return std::clamp((1.0 - idle_delta_ms / frame_ms) * 100.0, 0.0, 100.0);
}

PerfLogger::PerfLogger(const StartupPerf& startup) {
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

    // 先把启动阶段开销写成注释头。
    // 这样 CSV 既能被脚本读取，也能被人肉快速理解。
    std::fprintf(file_, "# sdl_setup_ms=%.3f\n", startup.sdl_setup_ms);
    std::fprintf(file_, "# scene_create_ms=%.3f\n", startup.scene_create_ms);
    std::fprintf(file_, "# bvh_build_ms=%.3f\n", startup.bvh_build_ms);
    std::fprintf(file_, "# upload_ms=%.3f\n", startup.upload_ms);
    std::fprintf(file_, "# denoiser_init_ms=%.3f\n", startup.denoiser_init_ms);
    std::fprintf(file_, "# display_resolution=%dx%d\n", startup.display_width, startup.display_height);
    std::fprintf(file_, "# render_resolution=%dx%d\n", startup.render_width, startup.render_height);
    // CSV 正文从这里开始。
    std::fprintf(file_,
                 "frame,presented,camera_moved,save_request,input_ms,render_ms,post_ms,present_ms,denoise_ms,total_ms,fps,"
                 "gt0_act_freq_mhz,gt1_act_freq_mhz,gt0_idle_delta_ms,gt1_idle_delta_ms,gt0_busy_pct,gt1_busy_pct,"
                 "npu_busy_pct,npu_freq_mhz,npu_mem_bytes,"
                 "dl_clamp,eh_clamp,eh_primary_clamp,eh_indirect_clamp,tp_diffuse,tp_specular,tp_refract,tp_rr,final_firefly_clamp,nan_or_inf\n");
    std::fflush(file_);

    std::cout << "[Perf] Logging frame timings to " << path_ << std::endl;
}

PerfLogger::~PerfLogger() {
    if (file_) {
        std::fflush(file_);
        std::fclose(file_);
    }
}

void PerfLogger::log_frame(int frame_index,
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

    // RenderStats 在默认模式下可能是空的。
    // 这里统一把“无统计”编码成 -1，避免 CSV 列数忽多忽少。
    auto stat_or_neg1 = [render_stats](uint32_t RenderStats::*member) -> int {
        if (!render_stats) {
            return -1;
        }
        return static_cast<int>(render_stats->*member);
    };

    // 一帧一行，方便后续用 Python / 表格软件做筛选和聚合。
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

    // 批量 flush，避免每帧都打断 IO 缓冲。
    ++pending_rows_;
    if (pending_rows_ >= kPerfFlushInterval) {
        std::fflush(file_);
        pending_rows_ = 0;
    }
}
