#include "pipeline.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>

// ======================================================================================
// 内部函数: 实际的 Worker 循环逻辑
// ======================================================================================
static void worker_loop(Pipeline* pipe) {
    while (true) {
        int frame_to_process = 0;

        // 1. 等待阶段 (Sleep)
        {
            std::unique_lock<std::mutex> lock(pipe->mtx);
            pipe->cv_worker.wait(lock, [pipe] {
                return pipe->quit || pipe->worker_busy;
            });

            if (pipe->quit) break;
            
            frame_to_process = pipe->current_frame;
        }

        // 2. 搬运阶段 (Device -> Host)
        cudaMemcpy(pipe->h_accum, pipe->d_staging, pipe->size_bytes, cudaMemcpyDeviceToHost);

        // 3. 计算阶段 (OpenMP Float -> Int)
        int w = pipe->width;
        int h = pipe->height;

        // 记录开始时间
        //auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
            Vec avg = pipe->h_accum[i] * (1.0f / frame_to_process);
            int r = toInt(avg.x);
            int g = toInt(avg.y);
            int b = toInt(avg.z);
            pipe->pixel_buffer[i] = (255 << 24) | (r << 16) | (g << 8) | b;
        }

        // 记录结束时间
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> elapsed = end - start;
        //if (frame_to_process % 100 == 0) fprintf(stderr, "Frame %d | Convert Time: %.2f ms\n", frame_to_process, elapsed.count());

        // 4. 完成阶段
        {
            std::lock_guard<std::mutex> lock(pipe->mtx);
            pipe->frame_ready = true;
            pipe->worker_busy = false; // 标记为空闲
        }
    }
}

// ======================================================================================
// API 实现
// ======================================================================================

void pipeline_init(Pipeline* pipe, Vec* h_accum, Vec* d_staging, uint32_t* pixel_buffer, int w, int h) {
    pipe->h_accum = h_accum;
    pipe->d_staging = d_staging;
    pipe->pixel_buffer = pixel_buffer;
    pipe->width = w;
    pipe->height = h;
    pipe->size_bytes = w * h * sizeof(Vec);
    
    pipe->quit = false;
    pipe->worker_busy = false;
    pipe->frame_ready = false;

    // 启动线程
    pipe->worker_thread = std::thread(worker_loop, pipe);
}

bool pipeline_try_dispatch(Pipeline* pipe, int current_gpu_frame) {
    std::lock_guard<std::mutex> lock(pipe->mtx);
    
    // 如果 Worker 正忙，就不打扰了，让 GPU 跑下一帧去吧
    if (pipe->worker_busy) {
        return false; 
    }

    // 如果 Worker 闲着，派活！
    pipe->current_frame = current_gpu_frame;
    pipe->worker_busy = true;
    pipe->cv_worker.notify_one(); // 唤醒
    return true;
}

bool pipeline_check_frame_ready(Pipeline* pipe) {
    std::lock_guard<std::mutex> lock(pipe->mtx);
    if (pipe->frame_ready) {
        pipe->frame_ready = false; // 消费掉这个标志
        return true;
    }
    return false;
}

void pipeline_destroy(Pipeline* pipe) {
    {
        std::lock_guard<std::mutex> lock(pipe->mtx);
        pipe->quit = true;
        pipe->worker_busy = true; // 确保条件满足以跳出 wait
    }
    pipe->cv_worker.notify_all();
    
    if (pipe->worker_thread.joinable()) {
        pipe->worker_thread.join();
    }
}
