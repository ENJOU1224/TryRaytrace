/*
 * ======================================================================================
 * [渲染器宿主程序] Renderer Host Implementation
 * ======================================================================================
 * 职责:
 * 1. 负责 CPU 到 GPU 的数据搬运 (Host -> Device)。
 * 2. 负责纹理文件的读取 (IO) 和 CUDA 纹理对象的创建。
 * 3. 负责启动 GPU 渲染内核。
 * 
 * 注意: 核心的 GPU 算法 (求交、着色) 已拆分至 "gpu_shading.cuh"。
 */

#include "gpu_context.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "loader.h"

// [关键]: 引入 GPU 核心逻辑头文件
// 这让我们可以在这里调用 __global__ 函数，同时保持代码整洁
#include "gpu_shading.cuh" 

// ======================================================================================
// 1. 全局显存指针 (Global Memory Pointers)
// ======================================================================================
// 这些变量存在于 CPU 内存中，但它们的值是 GPU 显存的地址。
// 它们充当了 CPU 和 GPU 共享数据的句柄。

// 场景物体数组 (不再受 Constant Memory 64KB 限制，可存数百万物体)
Object* d_objects_ptr = nullptr;

// BVH 树节点数组
LinearBVHNode* d_bvh_nodes = nullptr;

// 光源索引数组 (用于 NEE 采样)
int* d_light_indices = nullptr;
int d_light_count = 0;

// ======================================================================================
// 2. 纹理资源 (Texture Resources)
// ======================================================================================
// 纹理对象句柄非常小 (64-bit int)，且读取频率极高。
// 依然保留在 __constant__ 内存中以获得最佳缓存性能。

// #define MAX_TEXTURES 5
// __constant__ cudaTextureObject_t d_textures[MAX_TEXTURES]; 

// [辅助函数] 加载并上传纹理到 GPU
cudaTextureObject_t load_texture_to_gpu(const std::string& filename) {
    int w, h;
    unsigned char* rgb_data = load_ppm(filename.c_str(), &w, &h);
    if (!rgb_data) return 0;

    // 1. 格式对齐 (RGB -> RGBA)
    // 显卡硬件对 4 通道 (uchar4) 的访问效率远高于 3 通道。
    // 我们牺牲 25% 的显存空间，换取纹理采样性能。
    unsigned char* rgba_data = (unsigned char*)malloc(w * h * 4);
    for (int i = 0; i < w * h; i++) {
        rgba_data[i*4 + 0] = rgb_data[i*3 + 0]; // R
        rgba_data[i*4 + 1] = rgb_data[i*3 + 1]; // G
        rgba_data[i*4 + 2] = rgb_data[i*3 + 2]; // B
        rgba_data[i*4 + 3] = 255;               // A (完全不透明)
    }
    free(rgb_data); // 原始数据已无用

    // 2. 分配 CUDA Array
    // CUDA Array 是专门为纹理采样优化的不透明内存布局 (Block Linear)。
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, w, h);

    // 3. 数据上传
    const size_t spitch = w * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(cuArray, 0, 0, rgba_data, spitch, w * 4 * sizeof(unsigned char), h, cudaMemcpyHostToDevice);
    free(rgba_data);

    // 4. 创建纹理对象 (Texture Object)
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;   // U轴: 循环平铺
    texDesc.addressMode[1] = cudaAddressModeWrap;   // V轴: 循环平铺
    texDesc.filterMode = cudaFilterModeLinear;      // 过滤: 双线性插值 (平滑)
    texDesc.readMode = cudaReadModeNormalizedFloat; // 读取: 自动将 [0,255] 映射为 [0.0, 1.0]
    texDesc.normalizedCoords = 1;                   // 坐标: 使用归一化 UV [0,1]

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return texObj;
}

// ======================================================================================
// 3. 场景初始化接口 (Host API)
// ======================================================================================
// 这个函数由 main.cpp 调用，负责一次性将所有数据上传到显存。
void init_scene_data(const std::vector<Object>& objects, 
                     const std::vector<std::string>& texture_files,
                     const std::vector<LinearBVHNode>& nodes,
                     const std::vector<int>& light_indices) {
    
    // --- 1. 上传物体数据 (Global Memory) ---
    // 先释放旧显存 (支持热重载场景)
    if (d_objects_ptr) cudaFree(d_objects_ptr);
    
    size_t objects_size = objects.size() * sizeof(Object);
    if (objects_size > 0) {
        cudaMalloc(&d_objects_ptr, objects_size);
        cudaMemcpy(d_objects_ptr, objects.data(), objects_size, cudaMemcpyHostToDevice);
        printf("[Renderer] Uploaded %lu objects to Global Memory (%.2f KB).\n", 
               objects.size(), objects_size / 1024.0f);
    }

    // --- 2. 加载并上传纹理 ---
    std::vector<cudaTextureObject_t> temp_tex_objs;
    for (const auto& file : texture_files) {
        temp_tex_objs.push_back(load_texture_to_gpu(file));
    }
    
    // 将纹理句柄数组拷贝到常量内存
    if (!temp_tex_objs.empty()) {
        if (temp_tex_objs.size() > MAX_TEXTURES) {
            printf("[Renderer Warning] Texture count exceeds limit (%d).\n", MAX_TEXTURES);
        }
        size_t count = temp_tex_objs.size() < MAX_TEXTURES ? temp_tex_objs.size() : MAX_TEXTURES;
        cudaMemcpyToSymbol(d_textures, temp_tex_objs.data(), count * sizeof(cudaTextureObject_t));
    }

    // --- 3. 上传 BVH 树 ---
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    
    size_t nodes_size = nodes.size() * sizeof(LinearBVHNode);
    if (nodes_size > 0) {
        cudaMalloc(&d_bvh_nodes, nodes_size);
        cudaMemcpy(d_bvh_nodes, nodes.data(), nodes_size, cudaMemcpyHostToDevice);
        printf("[Renderer] Uploaded %lu BVH nodes.\n", nodes.size());
    }

    // --- 4. 上传光源索引 (用于 NEE) ---
    if (d_light_indices) cudaFree(d_light_indices);
    
    d_light_count = (int)light_indices.size();
    if (d_light_count > 0) {
        size_t lights_size = d_light_count * sizeof(int);
        cudaMalloc(&d_light_indices, lights_size);
        cudaMemcpy(d_light_indices, light_indices.data(), lights_size, cudaMemcpyHostToDevice);
        printf("[Renderer] Registered %d lights for NEE.\n", d_light_count);
    } else {
        printf("[Renderer Warning] No lights found! Scene will be dark.\n");
    }
}

// ======================================================================================
// 4. 内核启动接口 (Kernel Launcher)
// ======================================================================================
void launch_render_kernel(Vec* accum_buffer, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    // 计算网格维度
    dim3 threads(tx, ty);
    dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
    
    // 调用位于 gpu_shading.cuh 中的核心 Kernel
    // 我们将所有全局资源指针通过参数传递进去
    render_kernel_impl<<<blocks, threads>>>(
        accum_buffer, 
        width, height, 
        frame_seed, 
        cam, 
        d_bvh_nodes, 
        d_objects_ptr, 
        d_light_indices, 
        d_light_count
    );
    
    // 异步启动，无需 cudaDeviceSynchronize (由管线其他部分负责同步)
}
