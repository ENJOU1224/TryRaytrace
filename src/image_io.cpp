#include "image_io.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <omp.h> // 需要 OpenMP 加速转换

void save_snapshot(const Vec* h_accum, int w, int h, int frame, float focus_dist, float aperture) {
    // 1. 确保 logs 目录存在
    (void)system("mkdir -p logs");

    // 2. 生成文件名
    std::time_t now = std::time(nullptr);
    std::tm* t = std::localtime(&now);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", t);

    char filename[256];
    sprintf(filename, "logs/%s_Frame%d.ppm", time_str, frame);

    // 3. 转换数据 (OpenMP 加速)
    // 这里分配临时内存，用完即焚
    std::vector<unsigned char> img(w * h * 3);
    
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        // 显存数据已经是 Top-Down 的，直接线性拷贝
        Vec avg = h_accum[i] * (1.0f / frame);
        img[i * 3 + 0] = (unsigned char)toInt(avg.x);
        img[i * 3 + 1] = (unsigned char)toInt(avg.y);
        img[i * 3 + 2] = (unsigned char)toInt(avg.z);
    }

    // 4. 写入文件
    FILE* f = fopen(filename, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
        fwrite(img.data(), 1, w * h * 3, f);
        fclose(f);
        std::cout << "[IO] Snapshot saved: " << filename 
                  << " (Focus: " << focus_dist << ", Aperture: " << aperture << ")" << std::endl;
    } else {
        std::cerr << "[IO Error] Failed to write file: " << filename << std::endl;
    }
}
