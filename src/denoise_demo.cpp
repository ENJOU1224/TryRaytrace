#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>

#include "bvh.h"
#include "camera.h"
#include "common.h"
#include "denoiser.h"
#include "renderer.h"
#include "scene.h"

namespace {

constexpr int kDemoWidth = 1200;
constexpr int kDemoHeight = 800;
constexpr int kDemoFrameSeed = 1;
constexpr int kToggleIntervalMs = 2000;

bool g_quit = false;

void signal_handler(int) {
    g_quit = true;
}

// 这个演示程序的目标不是“交互式实时渲染”，而是“固定渲染一帧，然后肉眼对比降噪前后”。
// 因此它会比主程序简单很多：
// - 不处理相机移动
// - 不做持续累积
// - 只关心原图和降噪图切换展示

/**
 * 将单帧累积缓冲区整理成线性 RGB。
 * 这里不做 Tone Mapping，目的是让降噪器吃到未经显示压缩的线性数据。
 */
void build_linear_rgb_frame(const Vec* accum, int pixel_count, int frame_index, std::vector<float>& rgb_buffer) {
    for (int i = 0; i < pixel_count; ++i) {
        Vec color = accum[i] * (1.0f / frame_index);
        rgb_buffer[i * 3 + 0] = std::max(0.0f, color.x);
        rgb_buffer[i * 3 + 1] = std::max(0.0f, color.y);
        rgb_buffer[i * 3 + 2] = std::max(0.0f, color.z);
    }
}

/**
 * 将线性 RGB 转成 SDL 直接显示的 ARGB8888。
 * 这样演示窗口里显示的“原图”和“降噪图”都走完全一致的显示链路。
 */
void linear_rgb_to_argb8888(const float* rgb_buffer, int pixel_count, std::vector<uint32_t>& pixel_buffer) {
    for (int i = 0; i < pixel_count; ++i) {
        const float r = encode_display_value(rgb_buffer[i * 3 + 0]);
        const float g = encode_display_value(rgb_buffer[i * 3 + 1]);
        const float b = encode_display_value(rgb_buffer[i * 3 + 2]);
        pixel_buffer[i] = (255u << 24) |
                          (static_cast<uint32_t>(r * 255.0f + 0.5f) << 16) |
                          (static_cast<uint32_t>(g * 255.0f + 0.5f) << 8) |
                          static_cast<uint32_t>(b * 255.0f + 0.5f);
    }
}

std::vector<int> collect_light_indices(const Scene& scene) {
    std::vector<int> light_indices;
    for (size_t i = 0; i < scene.objects.size(); ++i) {
        if (scene.objects[i].emission.norm_len() > 0.1f) {
            light_indices.push_back(static_cast<int>(i));
        }
    }
    return light_indices;
}

void update_demo_title(SDL_Window* window,
                       bool denoise_ready,
                       bool show_denoised,
                       double render_ms,
                       double denoise_ms) {
    char title[256];
    if (denoise_ready) {
        std::snprintf(title,
                      sizeof(title),
                      "Denoise Demo | %s | Render %.1f ms | Denoise %.1f ms | 2 秒自动切换",
                      show_denoised ? "NPU 降噪图" : "原始单帧图",
                      render_ms,
                      denoise_ms);
    } else {
        std::snprintf(title,
                      sizeof(title),
                      "Denoise Demo | 仅原始单帧图 | Render %.1f ms | NPU 降噪不可用",
                      render_ms);
    }
    SDL_SetWindowTitle(window, title);
}

}  // namespace

int main() {
    std::signal(SIGINT, signal_handler);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[SDL Error] Initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Denoise Demo",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          kDemoWidth,
                                          kDemoHeight,
                                          SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             kDemoWidth,
                                             kDemoHeight);

    // 场景初始化流程和主程序保持一致，这样演示结果才具有可比性。
    Scene scene = create_cornell_box();
    BVH bvh;
    bvh.build(scene.objects);
    std::vector<int> light_indices = collect_light_indices(scene);
    init_scene_data(scene.objects, scene.texture_files, bvh.get_nodes(), light_indices);

    sycl::queue& q = get_renderer_queue();
    Vec* accum = sycl::malloc_shared<Vec>(kDemoWidth * kDemoHeight, q);
    std::memset(accum, 0, static_cast<size_t>(kDemoWidth) * kDemoHeight * sizeof(Vec));

    CameraController cam({50, 50, 295.6}, {0, 0, -1});
    CameraParams cam_params = cam.get_params(kDemoWidth, kDemoHeight);

    // 这里只渲染一次，所以 frame seed 固定为 1。
    auto render_begin = std::chrono::high_resolution_clock::now();
    launch_render_kernel(accum, kDemoWidth, kDemoHeight, kDemoFrameSeed, 16, 8, cam_params, nullptr);
    auto render_end = std::chrono::high_resolution_clock::now();
    const double render_ms =
        std::chrono::duration<double, std::milli>(render_end - render_begin).count();

    std::vector<float> raw_linear_rgb(static_cast<size_t>(kDemoWidth) * kDemoHeight * 3, 0.0f);
    std::vector<uint32_t> raw_pixels(static_cast<size_t>(kDemoWidth) * kDemoHeight, 0);
    build_linear_rgb_frame(accum, kDemoWidth * kDemoHeight, kDemoFrameSeed, raw_linear_rgb);
    linear_rgb_to_argb8888(raw_linear_rgb.data(), kDemoWidth * kDemoHeight, raw_pixels);

    // 然后在 CPU/NPU 侧对这一帧做一次同步降噪。
    FrameDenoiser denoiser;
    denoiser.initialize(kDemoWidth, kDemoHeight);

    bool denoise_ready = false;
    double denoise_ms = 0.0;
    std::vector<uint32_t> denoised_pixels(static_cast<size_t>(kDemoWidth) * kDemoHeight, 0);
    if (denoiser.is_enabled()) {
        auto denoise_begin = std::chrono::high_resolution_clock::now();
        denoise_ready = denoiser.run(raw_linear_rgb.data(), kDemoWidth, kDemoHeight);
        auto denoise_end = std::chrono::high_resolution_clock::now();
        denoise_ms =
            std::chrono::duration<double, std::milli>(denoise_end - denoise_begin).count();
        if (denoise_ready) {
            linear_rgb_to_argb8888(denoiser.output_rgb().data(),
                                   kDemoWidth * kDemoHeight,
                                   denoised_pixels);
        }
    }

    std::cout << "[Demo] 单帧渲染完成: " << render_ms << " ms" << std::endl;
    if (denoise_ready) {
        std::cout << "[Demo] NPU 降噪完成: " << denoise_ms << " ms" << std::endl;
        std::cout << "[Demo] 每 2 秒在“原始单帧图 / NPU 降噪图”之间自动切换。" << std::endl;
    } else {
        std::cout << "[Demo] NPU 降噪不可用，仅显示原始单帧图。" << std::endl;
        if (!denoiser.status().empty()) {
            std::cout << denoiser.status() << std::endl;
        }
    }
    std::cout << "[Demo] 按 ESC 或关闭窗口退出。" << std::endl;

    bool show_denoised = false;
    uint32_t last_toggle_ms = SDL_GetTicks();
    update_demo_title(window, denoise_ready, show_denoised, render_ms, denoise_ms);

    while (!g_quit) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                g_quit = true;
            } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
                g_quit = true;
            }
        }

        if (denoise_ready) {
            uint32_t now_ms = SDL_GetTicks();
            if (now_ms - last_toggle_ms >= static_cast<uint32_t>(kToggleIntervalMs)) {
                show_denoised = !show_denoised;
                last_toggle_ms = now_ms;
                update_demo_title(window, denoise_ready, show_denoised, render_ms, denoise_ms);
            }
        }

        // 根据当前切换状态选择显示原始图还是降噪图。
        const std::vector<uint32_t>& active_pixels =
            (denoise_ready && show_denoised) ? denoised_pixels : raw_pixels;
        SDL_UpdateTexture(texture, nullptr, active_pixels.data(), kDemoWidth * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    sycl::free(accum, q);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
