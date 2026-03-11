#include "async_denoiser.h"

#include <cstring>
#include <utility>

AsyncFrameDenoiser::AsyncFrameDenoiser() = default;

AsyncFrameDenoiser::~AsyncFrameDenoiser() {
    stop_worker();
}

void AsyncFrameDenoiser::initialize(int frame_width, int frame_height) {
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

    pending_frame_.assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);
    result_buffers_[0].assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);
    result_buffers_[1].assign(static_cast<size_t>(width_) * height_ * 3, 0.0f);

    denoiser_.initialize(width_, height_);
    status_ = denoiser_.status();
    if (!denoiser_.is_enabled()) {
        return;
    }

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

        if (!denoiser_.run(local_input.data(), width_, height_)) {
            status_ = denoiser_.status();
            enabled_.store(false);
            break;
        }

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
    stop_requested_.store(true);
    pending_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}
