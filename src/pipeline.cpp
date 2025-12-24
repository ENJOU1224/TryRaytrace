#include "pipeline.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <omp.h>      // OpenMP 多核并行库
#include <iostream>
#include <chrono>     // 用于高精度计时(调试用)

// ======================================================================================
// 内部函数: Worker 线程主循环
// ======================================================================================
// 这是一个"消费者"线程。它不断等待主线程("生产者")派发的数据处理任务。
// ======================================================================================
static void worker_loop(Pipeline* pipe) {
    while (true) {
        int frame_to_process = 0;

        // ------------------------------------------------------------------
        // 1. 等待阶段 (Wait & Sleep)
        // ------------------------------------------------------------------
        // 使用条件变量实现低功耗等待。
        // 如果没有任务，线程会挂起 (Sleep)，CPU 占用率为 0%。
        {
            std::unique_lock<std::mutex> lock(pipe->mtx);
            
            // wait() 会自动释放锁并阻塞，直到被 notify 且 lambda 返回 true
            pipe->cv_worker.wait(lock, [pipe] {
                // 唤醒条件: 收到退出信号 OR 收到新任务(worker_busy被置为true)
                return pipe->quit || pipe->worker_busy;
            });

            // 如果是退出信号，直接跳出循环
            if (pipe->quit) break;
            
            // 获取任务参数 (当前是第几帧)
            frame_to_process = pipe->current_frame;
        } 
        // 离开作用域，unique_lock 自动析构并解锁。
        // 此时 Worker 处于"忙碌"状态，主线程不会打扰我们。

        // ------------------------------------------------------------------
        // 2. 搬运阶段 (Data Transfer)
        // ------------------------------------------------------------------
        // 从 GPU 显存 (d_staging) 拷贝到 CPU 内存 (h_accum)。
        // 这里的 h_accum 是 Pinned Memory，传输走 DMA 通道，速度极快。
        cudaMemcpy(pipe->h_accum, pipe->d_staging, pipe->size_bytes, cudaMemcpyDeviceToHost);

        // ------------------------------------------------------------------
        // 3. 计算阶段 (Compute / Format Conversion)
        // ------------------------------------------------------------------
        // 将高精度 Float (累加值) 转换为 32位整数 ARGB (显示值)。
        // 这是一个计算密集型任务 (包含除法和 Gamma 校正)。
        
        int w = pipe->width;
        int h = pipe->height;

        // [OpenMP 优化]
        // schedule(static): 任务负载是完全均匀的，静态切分给每个核效率最高，调度开销最小。
        // 并行度: 会利用 CPU 的所有物理核和逻辑核。
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < w * h; i++) {
            // 平均化: 总能量 / 采样次数
            Vec avg = pipe->h_accum[i] * (1.0f / frame_to_process);
            
            // Gamma 校正 + 量化 (toInt 定义在 common.h)
            int r = toInt(avg.x);
            int g = toInt(avg.y);
            int b = toInt(avg.z);
            
            // 写入显示缓冲区 (格式: 0xAARRGGBB)
            pipe->pixel_buffer[i] = (255 << 24) | (r << 16) | (g << 8) | b;
        }

        // [调试代码] 性能分析 (平时注释掉)
        /*
        static auto last_time = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = now - last_time;
        if (frame_to_process % 100 == 0) {
            std::cout << "[Worker] Process time: " << diff.count() << " ms" << std::endl;
        }
        last_time = now;
        */

        // ------------------------------------------------------------------
        // 4. 完成阶段 (Notify)
        // ------------------------------------------------------------------
        // 告诉主线程：这一帧搞定了，可以拿去显示了。
        {
            std::lock_guard<std::mutex> lock(pipe->mtx);
            pipe->frame_ready = true;  // 标记产出就绪
            pipe->worker_busy = false; // 标记自身空闲 (可以接新活了)
        }
        // 这里不需要 notify 主线程，因为主线程是在渲染循环里轮询 check_frame_ready 的。
    }
}

// ======================================================================================
// API 实现
// ======================================================================================

// 初始化流水线
void pipeline_init(Pipeline* pipe, Vec* h_accum, Vec* d_staging, uint32_t* pixel_buffer, int w, int h) {
    // 绑定资源指针
    pipe->h_accum = h_accum;
    pipe->d_staging = d_staging;
    pipe->pixel_buffer = pixel_buffer;
    pipe->width = w;
    pipe->height = h;
    pipe->size_bytes = w * h * sizeof(Vec);
    
    // 初始化状态
    pipe->quit = false;
    pipe->worker_busy = false;
    pipe->frame_ready = false;

    // 启动后台线程
    pipe->worker_thread = std::thread(worker_loop, pipe);
}

// 尝试派发任务 (非阻塞)
// 返回值: true 表示派发成功，false 表示 Worker 正忙(丢帧)
bool pipeline_try_dispatch(Pipeline* pipe, int current_gpu_frame) {
    // 加锁检查状态
    std::lock_guard<std::mutex> lock(pipe->mtx);
    
    // 如果 Worker 正在处理上一帧，我们就跳过这一帧的显示更新
    // 策略: "Drop Frame" (丢弃中间帧)，保证 GPU 渲染不卡顿
    if (pipe->worker_busy) {
        return false; 
    }

    // Worker 闲着，派发新任务
    pipe->current_frame = current_gpu_frame;
    pipe->worker_busy = true;     // 标记为忙，防止重入
    pipe->cv_worker.notify_one(); // 唤醒沉睡的 Worker
    return true;
}

// 检查是否有新帧产出
bool pipeline_check_frame_ready(Pipeline* pipe) {
    // 加锁读取
    std::lock_guard<std::mutex> lock(pipe->mtx);
    
    if (pipe->frame_ready) {
        pipe->frame_ready = false; // 消费掉这个标志 (Reset)
        return true;
    }
    return false;
}

// 销毁流水线
void pipeline_destroy(Pipeline* pipe) {
    {
        std::lock_guard<std::mutex> lock(pipe->mtx);
        pipe->quit = true;        // 设置退出标志
        pipe->worker_busy = true; // 确保条件满足 (pipe->quit 为 true)，能跳出 wait
    }
    
    pipe->cv_worker.notify_all(); // 唤醒线程让它退出
    
    if (pipe->worker_thread.joinable()) {
        pipe->worker_thread.join(); // 等待线程安全结束
    }
}
