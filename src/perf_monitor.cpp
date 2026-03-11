#include "perf_monitor.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

constexpr int kPerfFlushInterval = 30;

}

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

double DeviceMetricsSampler::read_double(const char* path, double fallback) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return fallback;
    }

    double value = fallback;
    file >> value;
    return file.fail() ? fallback : value;
}

double DeviceMetricsSampler::compute_delta(double current, double previous) {
    if (current < 0.0 || previous < 0.0) {
        return -1.0;
    }
    double delta = current - previous;
    return delta >= 0.0 ? delta : -1.0;
}

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
