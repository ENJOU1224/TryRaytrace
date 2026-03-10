#include <csignal>
#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>

#include "common.h"     
#include "scene.h"      
#include "camera.h"     
#include "renderer.h"   
#include "pipeline.h"   
#include "input.h"      
#include "image_io.h"   
#include "bvh.h"        

std::atomic<bool> quit(false);
void signal_handler(int signal) { if (signal == SIGINT) quit = true; }

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    int w = 1200, h = 800;  
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* window = SDL_CreateWindow("SYCL Raytracer (Lunar Lake)", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);

    Scene scene = create_cornell_box();
    BVH bvh; bvh.build(scene.objects);

    std::vector<int> light_indices;
    for (size_t i = 0; i < scene.objects.size(); i++) {
        if (scene.objects[i].emission.norm_len() > 0.1f) light_indices.push_back((int)i);
    }

    init_scene_data(scene.objects, scene.texture_files, bvh.get_nodes(), light_indices);
    CameraController cam({50, 50, 295.6}, {0, 0, -1});

    sycl::queue q;
    try { q = sycl::queue(sycl::gpu_selector_v); } catch (...) { q = sycl::queue(sycl::cpu_selector_v); }

    Vec* d_accum = sycl::malloc_shared<Vec>(w * h, q);
    std::memset(d_accum, 0, w * h * sizeof(Vec));
    uint32_t* pixel_buffer = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    InputManager input; 
    int gpu_frame = 1;
    
    // FPS 统计
    auto last_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;

    while (!quit) {
        InputState state = input.process_events(cam);
        if (state.save_request) {
            save_snapshot(d_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture()); 
        }
        if (state.quit) quit = true;
        if (state.camera_moved) {
            gpu_frame = 1;
            std::memset(d_accum, 0, w * h * sizeof(Vec));
        }

        CameraParams cam_params = cam.get_params(w, h);
        launch_render_kernel(d_accum, w, h, gpu_frame, 16, 8, cam_params);
        
        #pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
            Vec color = d_accum[i] * (1.0f / gpu_frame);
            pixel_buffer[i] = (255 << 24) | (toInt(color.x) << 16) | (toInt(color.y) << 8) | toInt(color.z);
        }

        SDL_UpdateTexture(texture, NULL, pixel_buffer, w * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        
        // 统计 FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = current_time - last_time;
        last_time = current_time;
        fps = 0.9f * fps + 0.1f * (1.0f / delta.count()); // 平滑 FPS

        if (gpu_frame % 5 == 0) {
            char title[256];
            sprintf(title, "[%s] FPS: %.1f | Frame: %d | F: %.1f | A: %.2f", 
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    fps, gpu_frame, cam.get_focus_dist(), cam.get_aperture());
            SDL_SetWindowTitle(window, title);

            // --- [新增] 终端实时输出 ---
            printf("\r>> [SYCL] Rendering: Frame %d | FPS: %.1f", gpu_frame, fps);
            fflush(stdout);
        }
        gpu_frame++;
    }

    // 退出前自动保存最后一张图
    save_snapshot(d_accum, w, h, gpu_frame, cam.get_focus_dist(), cam.get_aperture()); 

    sycl::free(d_accum, q); free(pixel_buffer);
    SDL_DestroyTexture(texture); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit();
    return 0;
}
