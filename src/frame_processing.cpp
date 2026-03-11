#include "frame_processing.h"

#include <algorithm>

/**
 * 将路径追踪累积缓冲区转换为“线性 RGB 帧”。
 *
 * 注意这里故意不做 Gamma 校正：
 * 1. 大多数图像降噪模型更适合吃线性空间数据；
 * 2. Gamma 校正属于显示阶段，不应混入推理输入。
 */
void build_linear_rgb_frame(const Vec* accum, int pixel_count, int frame_index, std::vector<float>& rgb_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        // 当前缓冲里存的是“累计能量”，不是最终颜色。
        // 所以第一步必须先除以累计帧数，得到当前帧的平均估计值。
        Vec color = accum[i] * (1.0f / frame_index);

        // 这里先做最保守的安全处理：负值直接压到 0。
        // 不做 Tone Mapping，因为后面降噪器更适合吃线性空间数据。
        rgb_buffer[i * 3 + 0] = std::max(0.0f, color.x);
        rgb_buffer[i * 3 + 1] = std::max(0.0f, color.y);
        rgb_buffer[i * 3 + 2] = std::max(0.0f, color.z);
    }
}

/**
 * 将线性 RGB 图像转换成 SDL 纹理可直接显示的 ARGB8888。
 * 这里才执行 Tone Mapping + Gamma，保证“推理输入”和“显示输出”职责分离。
 */
void linear_rgb_to_argb8888(const float* rgb_buffer, int pixel_count, uint32_t* pixel_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        // 这里才做显示编码。
        // 这样可以保证：
        // - 降噪器看到的是线性颜色
        // - SDL 看到的是已经做过 Tone Mapping + Gamma 的显示颜色
        const float r = encode_display_value(rgb_buffer[i * 3 + 0]);
        const float g = encode_display_value(rgb_buffer[i * 3 + 1]);
        const float b = encode_display_value(rgb_buffer[i * 3 + 2]);
        pixel_buffer[i] = (255u << 24) |
                          (static_cast<uint32_t>(r * 255.0f + 0.5f) << 16) |
                          (static_cast<uint32_t>(g * 255.0f + 0.5f) << 8) |
                          static_cast<uint32_t>(b * 255.0f + 0.5f);
    }
}

/**
 * 纯渲染模式下的快速显示路径。
 *
 * 当 NPU 降噪关闭时，没有必要先把整帧写入线性 RGB 缓冲区，再做第二次遍历转成 ARGB。
 * 这里直接从累积缓冲区完成“平均化 -> Tone Mapping -> Gamma -> 打包显示”，
 * 可以少一次完整的 CPU 内存遍历和两块大缓冲区的常驻占用。
 */
void accum_to_argb8888(const Vec* accum, int pixel_count, int frame_index, uint32_t* pixel_buffer) {
    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        // 纯渲染模式下，直接从累积缓冲生成显示图像。
        // 这比“先落地到线性 RGB 中间缓冲，再转 ARGB”少一整次内存遍历。
        Vec color = accum[i] * (1.0f / frame_index);
        const float r = encode_display_value(color.x);
        const float g = encode_display_value(color.y);
        const float b = encode_display_value(color.z);
        pixel_buffer[i] = (255u << 24) |
                          (static_cast<uint32_t>(r * 255.0f + 0.5f) << 16) |
                          (static_cast<uint32_t>(g * 255.0f + 0.5f) << 8) |
                          static_cast<uint32_t>(b * 255.0f + 0.5f);
    }
}
