#include "async_denoiser.h"

#include <cstring>
#include <utility>

/**
 * 这个文件只做一件事：
 * 把同步的 FrameDenoiser 包装成一个简单的后台线程流水线。
 *
 * 主线程职责：
 * - 继续正常渲染
 * - 提交最新线性帧
 * - 显示最近完成的一帧降噪结果
 *
 * 后台线程职责：
 * - 等待最新待处理帧
 * - 调用同步 OpenVINO 推理
 * - 发布最新结果
 */

AsyncFrameDenoiser::AsyncFrameDenoiser() = default;

AsyncFrameDenoiser::~AsyncFrameDenoiser() {
    stop_worker();
}

void AsyncFrameDenoiser::initialize(int frame_width, int frame_height) {
    // 重新初始化前，先把旧线程完整停掉。
    // 这样可以避免：
    // - 旧线程仍在读写旧缓冲
    // - 重新初始化后出现双线程同时访问同一对象
    stop_worker();

    width_ = frame_width;
    height_ = frame_height;
    status_.clear();
    ready_buffer_index_.store(-1);
    ready_frame_id_.store(-1);
    pending_frame_id_ = -1;
    pending_ready_ = false;
    stop_requested_.store(false);
    enabled_.store(false);

    // 分配一块待处理缓冲和两块结果缓冲。
    // 结果缓冲做成双缓冲，是为了让“后台线程写结果”和“前台线程读结果”更少冲突。
    pending_frame_.assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);
    result_buffers_[0].assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);
    result_buffers_[1].assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);

    // 这里仍然复用同步版 FrameDenoiser，只是把它包进独立线程里。
    denoiser_.initialize(width_, height_);
    status_ = denoiser_.status();
    if (!denoiser_.is_enabled()) {
        return;
    }

    // 只有底层同步降噪器初始化成功，才真正起后台线程。
    enabled_.store(true);
    worker_ = std::thread(&AsyncFrameDenoiser::worker_loop, this);
}

bool AsyncFrameDenoiser::is_enabled() const {
    return enabled_.load();
}

const std::string& AsyncFrameDenoiser::status() const {
    return status_;
}

const std::string& AsyncFrameDenoiser::device_name() const {
    return denoiser_.device_name();
}

void AsyncFrameDenoiser::submit_frame(int frame_id, const float* input_rgb_nhwc) {
    if (!enabled_.load()) {
        return;
    }

    // 主线程这里采用“覆盖最新待处理帧”的策略，而不是排长队。
    // 好处：
    // 1. 不会因为 NPU 速度暂时跟不上而无限堆积内存。
    // 2. 显示端总是尽快追上最近的画面。
    const size_t byte_size = static_cast<size_t>(width_) * height_ * 3 * sizeof(float);
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        std::memcpy(pending_frame_.data(), input_rgb_nhwc, byte_size);
        pending_frame_id_ = frame_id;
        pending_ready_ = true;
    }
    pending_cv_.notify_one();
}

bool AsyncFrameDenoiser::has_result() const {
    return ready_buffer_index_.load() >= 0;
}

int AsyncFrameDenoiser::latest_result_frame_id() const {
    return ready_frame_id_.load();
}

const std::vector<float>& AsyncFrameDenoiser::latest_result() const {
    int index = ready_buffer_index_.load();
    if (index < 0) {
        static const std::vector<float> empty;
        return empty;
    }
    return result_buffers_[index];
}

void AsyncFrameDenoiser::worker_loop() {
    // local_input 是工作线程自己的本地缓冲。
    // 这样后台线程处理时，不会直接踩主线程正在写的 pending_frame。
    std::vector<float> local_input(static_cast<size_t>(width_) * height_ * 3, 0.0f);

    while (!stop_requested_.load()) {
        int frame_id = -1;
        {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            pending_cv_.wait(lock, [&]() {
                return stop_requested_.load() || pending_ready_;
            });

            if (stop_requested_.load()) {
                break;
            }

            // 只交换一次，把“最新待处理帧”转移到工作线程的本地缓冲。
            // 之前这里做了两次 swap，等价于把缓冲又换回去了，
            // 后台线程实际上一直在处理旧数据/空数据，显示上就会像是突然发黑。
            local_input.swap(pending_frame_);
            frame_id = pending_frame_id_;
            pending_ready_ = false;
        }

        // 这里是真正的同步 NPU 推理点，但它发生在后台线程里，
        // 所以不会阻塞主渲染循环。
        if (!denoiser_.run(local_input.data(), width_, height_)) {
            status_ = denoiser_.status();
            enabled_.store(false);
            break;
        }

        // 结果缓冲双缓冲写入：
        // - 一个缓冲当前可能正在被主线程显示
        // - 后台线程写另一个缓冲
        int current_ready = ready_buffer_index_.load();
        int write_index = (current_ready == 0) ? 1 : 0;
        std::memcpy(result_buffers_[write_index].data(),
                    denoiser_.output_rgb().data(),
                    static_cast<size_t>(width_) * height_ * 3 * sizeof(float));
        ready_frame_id_.store(frame_id);
        ready_buffer_index_.store(write_index);
    }
}

void AsyncFrameDenoiser::stop_worker() {
    // 析构或重新初始化时都要走这里，确保线程不会悬挂。
    stop_requested_.store(true);
    pending_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}
