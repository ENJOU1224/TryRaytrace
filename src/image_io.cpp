#include "image_io.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <omp.h> // [性能关键] OpenMP 库

// ======================================================================================
// 保存快照 (Save Snapshot)
// ======================================================================================
// 职责:
// 1. 将高精度的浮点累加数据 (Float RGB) 转换为 显示器用的字节数据 (Byte RGB)。
// 2. 执行 Gamma 校正 (这是一个耗时的数学运算)。
// 3. 将结果以 PPM (P6) 二进制格式写入硬盘。
// ======================================================================================
void save_snapshot(const Vec* h_accum, int w, int h, int frame, float focus_dist, float aperture) {
    
    // ------------------------------------------------------------------
    // 1. 准备环境
    // ------------------------------------------------------------------
    // 确保输出目录存在 (-p 表示如果存在就不报错，且自动创建父目录)
    // 这一步开销很小，且只在保存时触发，使用 system 调用最简单通用。
    (void)system("mkdir -p logs");

    // ------------------------------------------------------------------
    // 2. 生成文件名 (带时间戳和物理参数)
    // ------------------------------------------------------------------
    std::time_t now = std::time(nullptr);
    std::tm* t = std::localtime(&now);
    char time_str[64];
    // 格式: YYYY-MM-DD_HH-MM-SS
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", t);

    char filename[256];
    // [优化] 将焦距(F)和光圈(A)写进文件名，方便后续对比实验结果
    // 例如: logs/2025-12-25_12-00-00_Frame500_F110.0_A0.5.ppm
    sprintf(filename, "logs/%s_Frame%d_F%.1f_A%.2f.ppm", 
            time_str, frame, focus_dist, aperture);

    // ------------------------------------------------------------------
    // 3. 并行数据转换 (CPU Compute Bound)
    // ------------------------------------------------------------------
    // 为什么这里需要 OpenMP？
    // 虽然写硬盘是 IO 瓶颈，但在写硬盘之前，我们需要对 100万+ 个像素执行 toInt()。
    // toInt() 内部包含 pow(x, 1/2.2) 运算，这是非常昂贵的超越函数调用。
    // 如果单核跑，这里可能会卡顿 20-50ms。用 OpenMP 可以压到 5ms 以内。
    
    // 申请临时缓冲区 (栈上分配 vector 对象，堆上分配数据，RAII 自动释放)
    std::vector<unsigned char> img(w * h * 3);
    
    // schedule(static): 任务负载均衡，静态切分给每个核心，调度开销最小
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < w * h; i++) {
        // [平均化]: 累加值 / 采样次数
        Vec avg = h_accum[i] * (1.0f / frame);
        
        // [Gamma 校正 + 量化]
        // toInt 定义在 common.h 中
        unsigned char r = (unsigned char)toInt(avg.x);
        unsigned char g = (unsigned char)toInt(avg.y);
        unsigned char b = (unsigned char)toInt(avg.z);

        // 填充 RGB 数据
        img[i * 3 + 0] = r;
        img[i * 3 + 1] = g;
        img[i * 3 + 2] = b;
    }

    // ------------------------------------------------------------------
    // 4. 二进制写入 (Disk IO Bound)
    // ------------------------------------------------------------------
    FILE* f = fopen(filename, "wb"); // wb = Write Binary
    if (f) {
        // 写入 P6 文件头 (ASCII 文本)
        // 格式: P6 <宽> <高> <最大颜色值> <换行>
        fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
        
        // 写入像素数据 (二进制块)
        // fwrite 是缓冲 IO，但在写入大量数据时，它会直接调用系统调用，效率极高
        fwrite(img.data(), 1, w * h * 3, f);
        
        fclose(f);
        
        // 打印成功信息到终端
        std::cout << "[IO] Snapshot saved: " << filename << std::endl;
    } else {
        std::cerr << "[IO Error] Failed to open file for writing: " << filename << std::endl;
    }
    
    // vector img 会在这里自动释放内存
}
