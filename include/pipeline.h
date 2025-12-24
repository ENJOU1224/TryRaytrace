#pragma once
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "common.h" // 需要 Vec 定义

// ======================================================================================
// 流水线状态管理 (封装了所有线程同步细节)
// ======================================================================================
struct Pipeline {
    // --- 线程同步原语 ---
    std::mutex mtx;
    std::condition_variable cv_worker;
    
    // --- 状态标志 ---
    bool quit = false;
    bool worker_busy = false; // Worker 是否正在忙
    bool frame_ready = false; // 是否有新的一帧产出
    int current_frame = 0;    // 当前任务的帧号
    
    // --- 资源指针 (不拥有内存，只引用) ---
    Vec* h_accum = nullptr;        // Host 端 Pinned Memory
    Vec* d_staging = nullptr;      // Device 端显存快照
    uint32_t* pixel_buffer = nullptr; // 用于显示的像素数据
    int width = 0;
    int height = 0;
    size_t size_bytes = 0;

    // --- 线程对象 ---
    std::thread worker_thread;
};

// --- API 接口 ---

// 初始化并启动后台线程
void pipeline_init(Pipeline* pipe, Vec* h_accum, Vec* d_staging, uint32_t* pixel_buffer, int w, int h);

// 尝试派发任务 (非阻塞：如果 Worker 忙，则直接返回 false)
// 如果 Worker 空闲，则唤醒它处理 current_gpu_frame，并返回 true
bool pipeline_try_dispatch(Pipeline* pipe, int current_gpu_frame);

// 检查是否有新帧产出 (如果返回 true，说明 pixel_buffer 已更新)
bool pipeline_check_frame_ready(Pipeline* pipe);

// 销毁线程并清理
void pipeline_destroy(Pipeline* pipe);
