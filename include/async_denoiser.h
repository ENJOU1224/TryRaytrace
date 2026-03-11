#pragma once

#include "denoiser.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * @file async_denoiser.h
 * @brief NPU 异步降噪流水线
 *
 * 设计原则:
 * 1. GPU 渲染和 NPU 降噪并行，不让主渲染循环等待 NPU 推理完成。
 * 2. 使用“最新帧覆盖待处理帧”的简单策略，避免队列无限增长。
 * 3. 显示端始终读取最近一次完成的降噪结果，接受一定显示延迟。
 */
class AsyncFrameDenoiser {
public:
    AsyncFrameDenoiser();
    ~AsyncFrameDenoiser();

    AsyncFrameDenoiser(const AsyncFrameDenoiser&) = delete;
    AsyncFrameDenoiser& operator=(const AsyncFrameDenoiser&) = delete;

    void initialize(int frame_width, int frame_height);
    bool is_enabled() const;
    const std::string& status() const;
    const std::string& device_name() const;

    /**
     * @brief 提交一帧线性 RGB 到后台 NPU 线程
     *
     * 这里会复制输入帧到内部待处理缓冲区。
     * 如果上一帧还没开始处理，新提交的帧会覆盖旧的待处理帧，始终优先保留最新画面。
     */
    void submit_frame(int frame_id, const float* input_rgb_nhwc);

    bool has_result() const;
    int latest_result_frame_id() const;
    const std::vector<float>& latest_result() const;

private:
    void worker_loop();
    void stop_worker();

    int width_ = 0;
    int height_ = 0;
    FrameDenoiser denoiser_;

    std::atomic<bool> enabled_{false};
    std::atomic<bool> stop_requested_{false};
    std::string status_;

    // 后台工作线程：专门负责同步推理，避免主线程卡住。
    std::thread worker_;

    // pending_* 这一组变量表示“最新待处理帧”。
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::vector<float> pending_frame_;
    int pending_frame_id_ = -1;
    bool pending_ready_ = false;

    // ready_* 这一组变量表示“最近一帧已经推理完成、可安全显示的结果”。
    std::vector<float> result_buffers_[2];
    std::atomic<int> ready_buffer_index_{-1};
    std::atomic<int> ready_frame_id_{-1};
};
