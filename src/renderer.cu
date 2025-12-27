#include "renderer.h"
#include "aabb.h"
#include "bvh.h"
#include <curand_kernel.h> // CUDA 随机数生成库
#include <cstdio>          // 标准 IO (fopen, fscanf)
#include <cstring>         // 内存操作 (memset)
#include <string>

// ======================================================================================
// 全局显存数据 (Global GPU Memory)
// ======================================================================================

// 全局显存指针
// 这个指针本身存在 CPU 内存上，但它存储的地址指向 GPU 显存
Object* d_objects_ptr = nullptr;

// [纹理句柄数组]
// 存放由 CUDA Runtime API 创建的纹理对象 (Texture Object)。
// 这是一个 64 位的整数句柄，指向显存中实际的纹理数据。
#define MAX_TEXTURES 5
__constant__ cudaTextureObject_t d_textures[MAX_TEXTURES]; 

// [新增] BVH 节点数据 (Global Memory)
// 节点数量可能很多，常量内存放不下，所以用普通显存指针
LinearBVHNode* d_bvh_nodes = nullptr;

// [新增] 光源索引数组
int* d_light_indices = nullptr;
int d_light_count = 0; // 这是一个 Host 端的变量，但在 Kernel 启动时传进去，或者拷贝到 Symbol

// ======================================================================================
// 辅助工具: 手写 P6 图片加载器 (Host CPU)
// ======================================================================================
// 为什么要手写？为了零依赖。
// 这是一个标准的 PPM (P6 二进制格式) 解析器。
unsigned char* load_ppm(const char* filename, int* w, int* h) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("[Texture Error] Cannot open file: %s\n", filename);
        return NULL;
    }

    char header[64];
    // 读取文件头 "P6"
    if (fscanf(fp, "%63s", header) != 1) { /* 忽略警告 */ }
    
    if (strcmp(header, "P6") != 0) {
        printf("[Texture Error] Not a P6 binary PPM: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    int max_val;
    // 读取 宽、高、最大亮度
    if (fscanf(fp, "%d %d %d", w, h, &max_val) != 3) { /* 忽略警告 */ }
    
    // [关键细节]: fscanf 读完数字后，文件指针停在换行符前。
    // 必须吃掉这个换行符，否则它会被当成像素数据的第一个字节。
    fgetc(fp); 

    // 分配内存 (RGB 3通道)
    size_t bytes = (size_t)(*w) * (*h) * 3;
    unsigned char* data = (unsigned char*)malloc(bytes);
    
    // 读取二进制像素数据
    if (fread(data, 1, bytes, fp) != bytes) {
        printf("[Texture Error] Unexpected EOF: %s\n", filename);
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    printf("[Texture] Loaded: %s (%dx%d)\n", filename, *w, *h);
    return data;
}

// [纹理上传函数]
// 负责读取文件、转换格式、分配显存、创建纹理对象
cudaTextureObject_t load_texture_to_gpu(const std::string& filename) {
    int w, h;
    unsigned char* rgb_data = load_ppm(filename.c_str(), &w, &h);
    if (!rgb_data) return 0;

    // [格式转换]: RGB -> RGBA
    // CUDA 的纹理硬件对 4 通道数据 (uchar4) 的支持最好，访问效率最高 (对齐访问)。
    // 所以我们手动把 RGB 扩展为 RGBA，Alpha 设为 255。
    unsigned char* rgba_data = (unsigned char*)malloc(w * h * 4);
    for (int i = 0; i < w * h; i++) {
        rgba_data[i*4 + 0] = rgb_data[i*3 + 0];
        rgba_data[i*4 + 1] = rgb_data[i*3 + 1];
        rgba_data[i*4 + 2] = rgb_data[i*3 + 2];
        rgba_data[i*4 + 3] = 255; // Alpha
    }
    free(rgb_data); // 释放原始数据

    // [显存分配]: CUDA Array
    // cudaMallocArray 分配的是专门针对纹理采样的显存布局 (Block Linear Layout)，
    // 比普通的线性内存 (Pitch Linear) 有更好的 2D 局部性缓存命中率。
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, w, h);

    // [数据拷贝]: Host -> Device Array
    const size_t spitch = w * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(cuArray, 0, 0, rgba_data, spitch, w * 4 * sizeof(unsigned char), h, cudaMemcpyHostToDevice);
    free(rgba_data);

    // [资源描述符]: 告诉 CUDA 数据在哪里
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // [纹理描述符]: 告诉 CUDA 怎么采样
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;   // 循环平铺 (Wrap)
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;      // 双线性插值 (Bilinear Filter)
    texDesc.readMode = cudaReadModeNormalizedFloat; // 自动归一化: [0, 255] -> [0.0, 1.0]
    texDesc.normalizedCoords = 1;                   // 使用归一化坐标: UV [0.0, 1.0]

    // [创建对象]
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

// ======================================================================================
// 宿主端接口 (Host Wrapper)
// ======================================================================================
void init_scene_data(const std::vector<Object>& objects, 
                     const std::vector<std::string>& texture_files,
                     const std::vector<LinearBVHNode>& nodes,
                     const std::vector<int>& light_indices) {
    // 1. 上传物体数据到全局内存
    int count = objects.size();
    size_t objects_size = count * sizeof(Object);

    // 如果之前分配过，先释放
    if (d_objects_ptr) cudaFree(d_objects_ptr);

    // 申请显存 (不再受 64KB 限制，显存有多大就能申请多大)
    cudaMalloc(&d_objects_ptr, objects_size);
    cudaMemcpy(d_objects_ptr, objects.data(), objects_size, cudaMemcpyHostToDevice);

    printf("[Renderer] Uploaded %d objects to Global Memory (%.2f KB).\n", 
           count, objects_size / 1024.0f);

    // 2. 加载并上传纹理
    std::vector<cudaTextureObject_t> temp_tex_objs;
    for (const auto& file : texture_files) {
        temp_tex_objs.push_back(load_texture_to_gpu(file));
    }
    
    // 3. 上传纹理句柄数组
    if (!temp_tex_objs.empty()) {
        cudaMemcpyToSymbol(d_textures, temp_tex_objs.data(), temp_tex_objs.size() * sizeof(cudaTextureObject_t));
    }
    // 3. [新增] 上传 BVH 节点
    // 先释放旧的 (如果有)
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    
    size_t nodes_size = nodes.size() * sizeof(LinearBVHNode);
    cudaMalloc(&d_bvh_nodes, nodes_size);
    cudaMemcpy(d_bvh_nodes, nodes.data(), nodes_size, cudaMemcpyHostToDevice);
    
    printf("[Renderer] Uploaded %lu BVH nodes to GPU.\n", nodes.size());

    // 4. [新增] 上传光源索引
    if (d_light_indices) cudaFree(d_light_indices);
    d_light_count = light_indices.size();
    
    if (d_light_count > 0) {
        size_t lights_size = d_light_count * sizeof(int);
        cudaMalloc(&d_light_indices, lights_size);
        cudaMemcpy(d_light_indices, light_indices.data(), lights_size, cudaMemcpyHostToDevice);
        printf("[Renderer] Lights registered: %d\n", d_light_count);
    } else {
        printf("[Renderer Warning] No lights found in scene!\n");
    }
}


// [辅助函数] 计算三角形面积
__device__ float triangle_area(const Object& obj) {
    Vec e1 = obj.v1 - obj.v0;
    Vec e2 = obj.v2 - obj.v0;
    return e1.cross(e2).norm_len() * 0.5f; 
}

// ============================================================================
// PBR 数学辅助函数
// ============================================================================

// [Schlick 近似]: 计算菲涅尔反射率
// cosine: 视角与法线夹角
// F0: 垂直入射时的反射率 (金属为 Albedo，非金属通常为 0.04)
__device__ Vec fresnel_schlick(float cosine, Vec F0) {
    // F = F0 + (1-F0) * (1 - cos)^5
    return F0 + (make_vec(1.0f,1.0f,1.0f) - F0) * powf(1.0f - cosine, 5.0f);
}

// [粗糙反射]: 生成以 perfect_refl 为中心的随机向量
__device__ Vec sample_rough_reflection(Vec perfect_refl, float roughness, curandState* state) {
    // 简单的球形扰动模型
    float r1 = curand_uniform(state) * 2.0f * M_PI;
    float r2 = curand_uniform(state);
    // roughness 映射为扰动半径 (0~1)
    float radius = roughness; 
    
    // 生成单位圆盘上的随机点
    // 注意：这里为了性能用了简化算法，严谨的 PBR 应该用 GGX 重要性采样
    float disk_r = radius * sqrtf(r2); // 均匀分布
    Vec random_vec = make_vec(disk_r * cosf(r1), disk_r * sinf(r1), 1.0f - disk_r); // 粗略朝向 Z
    
    // 构建局部坐标系将 random_vec 转换到 perfect_refl 空间 (略，直接简单相加归一化)
    // 这种简单相加法在 roughness 很大时不太准确，但够用了
    // 生成一个随机单位球向量
    float z = 1.0f - 2.0f * r2;
    float r = sqrtf(1.0f - z * z);
    Vec random_sphere = make_vec(r * cosf(r1), r * sinf(r1), z);
    
    return (perfect_refl + random_sphere * roughness).norm();
}

// ======================================================================================
// 设备端核心算法 (Device Kernel)
// ======================================================================================

// [函数]: 计算光线与物体的交点
// 返回值: t (距离)。如果未相交则返回 0.0。
__device__ float intersect(const Object& obj, const Vec& r_o, const Vec& r_d) {
    float eps = 1e-5f; // 防止自遮挡的微小偏移量
    
      // 1. 计算两边向量
      Vec e1 = obj.v1 - obj.v0;
      Vec e2 = obj.v2 - obj.v0;
      
      // 2. 计算行列式 (用于判断光线是否平行于三角形平面)
      Vec h = r_d.cross(e2);
      float a = e1.dot(h);
      
      // 如果 a 接近 0，说明光线平行，未相交
      if (a > -eps && a < eps) return 0.0f;
      
      float f = 1.0f / a;
      Vec s = r_o - obj.v0;
      
      // 3. 计算重心坐标 u
      float u = f * s.dot(h);
      if (u < 0.0f || u > 1.0f) return 0.0f; // 超出三角形范围
      
      Vec q = s.cross(e1);
      
      // 4. 计算重心坐标 v
      float v = f * r_d.dot(q);
      if (v < 0.0f || u + v > 1.0f) return 0.0f; // 超出三角形范围
      
      // 5. 计算距离 t
      float t = f * e2.dot(q);

      // 只有当 t > eps (在光线前方) 才算有效交点
      if (t > eps) return t;
      return 0.0f;     
}

// [阴影光线测试]
// 检查从 origin 到 target 之间是否有遮挡
// max_dist: origin 到 target 的距离 (我们会稍微减一点点以防止打中 target 自己)
__device__ bool trace_shadow(const Vec& origin, const Vec& dir, float max_dist, 
                             LinearBVHNode* nodes, Object* scene_objects) {
    
    Vec r_inv_d = make_vec(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    
    int stack[32];
    int ptr = 0;
    stack[ptr++] = 0;

    while (ptr > 0) {
        int idx = stack[--ptr];
        LinearBVHNode node = nodes[idx];

        // AABB 测试
        // 注意：t_max 设为 max_dist。如果盒子比灯还远，就不需要检测了。
        if (!node.bounds.hit(origin, r_inv_d, 0.001f, max_dist)) continue;

        if (node.is_leaf) {
            for (int k = 0; k < node.primitive_count; k++) {
                int obj_idx = node.primitive_offset + k;
                const Object& obj = scene_objects[obj_idx];
                
                // 这里我们要忽略光源自己 (虽然 max_dist 已经限制了，但为了保险)
                // 简单的做法是：如果撞到的物体是发光的，我们认为它是光源，不构成遮挡
                // 但更严谨的做法是依靠距离判断。
                
                float t = intersect(obj, origin, dir);
                
                // [关键] 只要发现一个物体挡在中间 (0.001 < t < max_dist)
                // 立即返回 true (有遮挡)
                if (t > 0.001f && t < max_dist - 0.001f) {
                    return true; // 被挡住了！
                }
            }
        } else {
            stack[ptr++] = node.right_child_idx;
            stack[ptr++] = node.left_child_idx;
        }
    }
    
    return false; // 一路通畅，没遮挡
}

// [内核]: 路径追踪主循环
__global__ void render_kernel_impl(Vec* accum_buffer, int width, int height, int frame_seed, CameraParams cam, LinearBVHNode* bvh_nodes, Object* scene_objects, int* light_indices, int light_count) {
    // 1. 线程索引与边界检查
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = (height - y - 1) * width + x;

    // 2. 随机数初始化
    curandState state;
    curand_init(1984 + frame_seed, i, 0, &state);

    // 3. 相机光线生成 (包含景深/薄透镜模型)
    // -----------------------------------------------------------
    // 抗锯齿: 在像素内随机抖动
    float r1 = 2 * curand_uniform(&state);
    float r2 = 2 * curand_uniform(&state);
    float dx = r1 < 1 ? sqrtf(r1) - 1 : 1 - sqrtf(2 - r1);
    float dy = r2 < 1 ? sqrtf(r2) - 1 : 1 - sqrtf(2 - r2);
    
    // 计算理想针孔方向
    Vec dir_pinhole = (cam.cx * (((x + .5f + dx) / width - .5f)) + 
                       cam.cy * (((y + .5f + dy) / height - .5f)) + cam.dir).norm();
    
    // 景深: 透镜采样
    Vec lens_offset = {0, 0, 0};
    if (cam.lens_radius > 0.0f) {
        float lr = cam.lens_radius * sqrtf(curand_uniform(&state));
        float ltheta = 2 * M_PI * curand_uniform(&state);
        Vec u = cam.cx.norm();
        Vec v = cam.cy.norm();
        lens_offset = u * (lr * cosf(ltheta)) + v * (lr * sinf(ltheta));
    }

    // 计算实际光线
    // 焦点 = 相机位置 + 针孔方向 * 焦距
    Vec p_focus = cam.pos + dir_pinhole * cam.focus_dist;
    // 起点 = 相机位置 + 透镜偏移
    Vec r_o = cam.pos + lens_offset;
    // 方向 = 起点 -> 焦点
    Vec r_d = (p_focus - r_o).norm();


    // 4. 路径追踪循环
    // -----------------------------------------------------------
    Vec throughput = make_vec(1, 1, 1);
    Vec radiance = make_vec(0, 0, 0);
    const int MAX_DEPTH = 10;
    const int RR_THRESHOLD = 3;
    Refl_t prev_refl_mode = SPEC;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // -----------------------------------------------------------
        // [BVH 遍历] Stackless / Stack-based Traversal
        // -----------------------------------------------------------
        auto safe_inv = [](float x) { 
            return (fabsf(x) < 1e-8f) ? (x >= 0 ? 1e20f : -1e20f) : (1.0f / x); 
        };

        Vec r_inv_d = make_vec(
            safe_inv(r_d.x), 
            safe_inv(r_d.y), 
            safe_inv(r_d.z)
        );
        float d_min = 1e20f;
        int id = -1;

        // 模拟递归的栈
        // 深度 32 足够遍历几十万面的场景 (2^32)
        int stack[32];
        int stack_ptr = 0; 

        // 将根节点 (索引0) 压栈
        stack[stack_ptr++] = 0; // 压入根节点

        while (stack_ptr > 0) {
            // 弹出栈顶节点
            int node_idx = stack[--stack_ptr];

            // 从 Global Memory 读取节点数据
            // 这里的读取会有延迟，但 GPU 会通过切换线程来隐藏它
            LinearBVHNode node = bvh_nodes[node_idx];

            // 1. 检查光线是否击中该节点的包围盒 (AABB)
            // 注意: 我们把当前的 d_min 传进去作为 t_max。
            // 如果盒子虽然打中了，但在已知交点及其后面，那也不用算了 (遮挡剔除)。
            if (!node.bounds.hit(r_o, r_inv_d, 0.0f, d_min)) continue;

            // 2. 叶子节点: 遍历物体
            if (node.is_leaf) {
                // 注意: node.primitive_count 可能不止 1 个
                for (int i = 0; i < node.primitive_count; i++) {
                    int obj_idx = node.primitive_offset + i;
                    // 求交
                    float t = intersect(scene_objects[obj_idx], r_o, r_d);
                    // 如果打中了，且比当前最近的还要近
                    if (t > 0.0f && t < d_min) {
                        d_min = t;
                        id = obj_idx;
                    }
                }
            } 
            // 3. 如果是内部节点 -> 将左右孩子压栈
            else {
                // 简单的压栈顺序：先压右，再压左 (这样下一次循环先处理左)
                // 进阶优化是根据光线方向决定先访问哪个孩子 (Front-to-Back)，这里先用简单版
                stack[stack_ptr++] = node.right_child_idx;
                stack[stack_ptr++] = node.left_child_idx;
            }
        }

        if (id < 0) break; // Miss


        // --- 遍历结束 ---
        // 此时 d_min 存的就是全场景最近交点，id 是物体索引


        const Object& obj = scene_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        
        // --- 法线计算 (几何属性) ---
        // [三角形法线]: 两条边做叉积
        // 注意: 这里我们使用"面法线" (Face Normal)，整个三角形是平的。
        // 如果要平滑渲染，需要重心坐标插值顶点法线 (Vertex Normal)，但那是进阶内容。
        Vec e1 = obj.v1 - obj.v0;
        Vec e2 = obj.v2 - obj.v0;
        Vec n = e1.cross(e2).norm();

        // 2. [新增] 如果启用了平滑，计算插值法线
        if (obj.use_smooth) {
            // 需要重心坐标 (u, v, w)
            // 既然 intersect 只返回了 t，我们这里为了省显存带宽不存 u/v，直接重算一遍
            // (MT算法很快，重算比读取显存更值)
            
            // ... (复制 Möller–Trumbore 的前半部分算 u, v) ...
            Vec h = r_d.cross(e2);
            float a = e1.dot(h);
            float f = 1.0f / a;
            Vec s = r_o - obj.v0;
            float u = f * s.dot(h);
            Vec q = s.cross(e1);
            float v = f * r_d.dot(q);
            float w = 1.0f - u - v;

            // 插值公式: N = w*n0 + u*n1 + v*n2
            n = obj.vn0 * w + obj.vn1 * u + obj.vn2 * v;
            n.norm(); // 插值后长度可能变短，必须重新归一化
        }


        // 确保法线朝向观察者 (双面渲染)
        Vec nl = n.dot(r_d) < 0 ? n : n * -1;
        
        // // [调试模式] 强制输出法线颜色
        // 把 (-1, 1) 的法线映射到 (0, 1) 的颜色
        // radiance = (n + make_vec(1.0f, 1.0f, 1.0f)) * 0.5f; 
        //
        // accum_buffer[i] = accum_buffer[i] + radiance;
        // return; // <--- 强制结束，只看第一帧的几何信息
        
        // =========================================================
        // [材质属性获取] - 这里是关键
        // =========================================================
        Vec albedo = obj.albedo; // 基础色
        float metallic = obj.metallic;
        float roughness = obj.roughness;
        float transmission = obj.transmission;

        // --- 1. 纹理采样 (Planar Mapping) ---
        // 即使是三角形，如果 tex_id >= 0，我们强行用平面映射贴图
        if (obj.tex_id >= 0) {
            const float scale = 0.01f;
            float u, v;

            // 简单根据法线朝向决定投影平面
            if (fabsf(n.y) > 0.9f)      { u = x_hit.x; v = x_hit.z; }
            else if (fabsf(n.x) > 0.9f) { u = x_hit.z; v = x_hit.y; }
            else                        { u = x_hit.x; v = x_hit.y; }
            
            u *= scale; v *= scale;
            v = 1.0f - v; // 翻转 V

            // 硬件采样
            float4 tex = tex2D<float4>(d_textures[obj.tex_id], u, v);
            // 纹理颜色 乘以 基础颜色
            albedo = albedo.mult(make_vec(tex.x, tex.y, tex.z));
        }

        // --- 2. 自发光累加 ---
        // [NEE 修正]: 
        // 1. 如果是第一帧 (depth==0)，必须加 (直接看灯)。
        // 2. 如果上一次是镜面反射 (SPEC)，必须加 (因为 NEE 不处理镜面)。
        // 3. 如果上一次是漫反射 (DIFF)，不能加！(因为 NEE 已经算过这盏灯对漫反射的贡献了)。
        
        bool is_specular_bounce = (prev_refl_mode == SPEC) || (prev_refl_mode == REFR);
        
        if (is_specular_bounce) {
            // if(depth >= 2)radiance = radiance + throughput.mult(obj.emission).mult({99,99,99}) ;
            // else radiance = radiance + throughput.mult(obj.emission).mult({0.1,0.1,0.1}) ;
            radiance = radiance + throughput.mult(obj.emission) ;
        }

        if (obj.emission.x > 0.001f || obj.emission.y > 0.001f || obj.emission.z > 0.001f) {
            break; 
        }

        // =========================================================
        // [PBR 光路选择: 基于能量的重要性采样]
        // =========================================================
        
        // 1. 准备非线性参数
        // [金属度曲线]: 只要有一点金属度，就大幅抑制漫反射。
        // 使用平滑阶梯或者幂函数。这里用幂函数让它更"硬"。
        // 0.5 -> 0.06, 0.1 -> 0.0001
        float diffuse_suppression = powf(1.0f - metallic, 2.0f); 
        
        // [粗糙度曲线]: 粗糙度越高，镜面反射的"视觉权重"下降得越快。
        // 我们用这个系数来衰减 Specular 的采样概率。
        // 0.0 -> 1.0 (全反射), 1.0 -> 0.0 (无反射)
        float spec_attenuation = 1.0f - (roughness * roughness);
        if (spec_attenuation < 0.0f) spec_attenuation = 0.0f;

        // 2. 计算菲涅尔项 (Schlick)
        Vec F0 = make_vec(0.04f, 0.04f, 0.04f);
        F0 = F0 * (1.0f - metallic) + albedo * metallic;
        
        // 使用反向光线计算视角
        float cos_theta = fmaxf(nl.dot(r_d * -1.0f), 0.0f);
        Vec F = fresnel_schlick(cos_theta, F0);
        float F_avg = (F.x + F.y + F.z) / 3.0f;

        // 3. 计算能量权重 (Weights)
        
        // 镜面反射权重: 
        // 基础是菲涅尔(F_avg)。
        // 乘以 粗糙度衰减系数 (spec_attenuation)。
        // 结果: 粗糙墙面的 w_spec 会变得极小，概率极低。
        float w_spec = F_avg * spec_attenuation;

        // 透射权重:
        // 基础是 (1-F) * transmission。
        // 同样受粗糙度影响 (可选，磨砂玻璃透光性也会变差/变散)。
        float w_trans = (1.0f - F_avg) * transmission;
        
        // 漫反射权重:
        // 基础是 (1-F) * (1-transmission)。
        // 乘以 金属抑制系数 (diffuse_suppression)。
        // 乘以 Albedo 亮度 (黑色的东西漫反射概率低)。
        float albedo_lum = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
        float w_diff = (1.0f - F_avg) * (1.0f - transmission) * diffuse_suppression * albedo_lum;

        // 4. 归一化为概率 (PDF)
        float sum = w_spec + w_trans + w_diff;
        
        // 防止除以零 (如果全黑且全吸收)
        if (sum < 1e-5f) {
            // 给一个默认值，或者结束
            w_diff = 1.0f; sum = 1.0f; 
        }

        float p_spec = w_spec / sum;
        float p_trans = w_trans / sum;

        // --- 3. 俄罗斯轮盘赌 ---
        if (depth > RR_THRESHOLD) {
            // 用 albedo 的最大分量作为存活概率
            float p = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
            if (p < 0.05f) p = 0.05f; // 最低存活率
            if (curand_uniform(&state) < p) throughput = throughput * (1.0f / p);
            else break;
        }

        float rnd = curand_uniform(&state);

        // --- 分支 A: 镜面反射 (Specular) ---
        // 概率: p_spec (由菲涅尔决定)
        if (rnd < p_spec) {

            Vec perfect = r_d - n * 2 * n.dot(r_d);
            r_d = sample_rough_reflection(perfect, roughness, &state);
            
            // 检查是否射入内部 (被微表面遮挡)
            if (r_d.dot(nl) <= 0.0f) break; 

            // 能量更新: 
            // 理论上 throughput *= F。但为了蒙特卡洛平衡，要除以概率 p_spec。
            // F / p_spec 近似为 1 (如果是白色 F)。
            // 严谨写法: throughput = throughput * F * (1.0f / p_spec);
            // 简单写法 (假设 F 是颜色):
            float weight = 1.0f / p_spec;
            throughput = throughput.mult(F) * weight;

            r_o = x_hit + nl * 1e-3f;
            prev_refl_mode = SPEC;
        }
        // --- 分支 B: 透射 (Transmission / Glass) ---
        // 概率: (1 - p_spec) * transmission
        else if (rnd < p_spec + p_trans) {
            // 1. 准备物理参数
            bool into = n.dot(nl) > 0; // 检查是射入还是射出
            float nc = 1.0f;           // 空气 IOR
            float nt = obj.ior;        // 物体 IOR (在 scene.h 里定义的, 通常1.45-1.5)
            float nnt = into ? nc / nt : nt / nc;
            float ddn = r_d.dot(nl);
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

            // 2. 全内反射 (Total Internal Reflection) 检测
            // 如果 cos2t < 0，说明角度太刁钻，光出不去，必须反射回内部
            if (cos2t < 0.0f) {
                Vec perfect_refl = r_d - n * 2.0f * n.dot(r_d);
                r_d = sample_rough_reflection(perfect_refl, roughness, &state);
                
                // 依然在物体内部，所以沿着法线往里推 (或者沿着新方向推)
                r_o = x_hit + r_d * 1e-4f;
            } 
            // 3. 正常折射 (Refraction)
            else {
                // 斯涅尔定律计算折射方向
                Vec tdir = (r_d * nnt - n * ((into ? 1.0f : -1.0f) * (ddn * nnt + sqrtf(cos2t)))).norm();

                // 粗糙透射 (Rough Transmission): 磨砂玻璃效果
                // 我们直接复用 sample_rough_reflection 的逻辑，但以透射方向为中心
                if (roughness > 0.0f) {
                    // 生成扰动向量
                    float r1 = curand_uniform(&state) * 2.0f * M_PI;
                    float r2 = curand_uniform(&state);
                    float radius = roughness;
                    float z = 1.0f - 2.0f * r2; // z in [-1, 1]
                    float r = sqrtf(1.0f - z * z);
                    
                    // 这是一个随机单位球向量
                    Vec random_vec = make_vec(r * cosf(r1), r * sinf(r1), z);
                    
                    // 混合并归一化
                    tdir = (tdir + random_vec * roughness).norm();
                }
                
                r_d = tdir;
                
                // [关键]: 穿过界面，所以要沿着折射方向往前推
                r_o = x_hit + r_d * 1e-4f;
            }

            // 4. 能量/颜色计算 (Monte Carlo Weight)
            // 玻璃的颜色通常是白色的，但如果 albedo 有颜色，代表它滤光。
            // 必须除以这个分支的概率 PDF，否则画面会偏暗。
            float p_branch = (1.0f - p_spec) * transmission;
            
            // 防止除以 0
            if (p_branch > 1e-4f) {
                throughput = throughput.mult(albedo) * (1.0f / p_branch);
            }
            prev_refl_mode = REFR; 
        }
        // --- 分支 C: 漫反射 (Diffuse) ---
        // 概率: 剩余部分
        else {

            // -----------------------------------------------------
            // [NEE] 显式光源采样
            // -----------------------------------------------------
            // 只有当表面是漫反射时，NEE 收益才大。
            if (light_count > 0) {
                // 1. 随机选灯
                int l_idx = (int)(curand_uniform(&state) * (light_count - 0.001f));
                const Object& light = scene_objects[light_indices[l_idx]]; // 注意用 scene_objects 指针

                // 2. 在灯上选点 (三角形均匀采样)
                float r1 = curand_uniform(&state);
                float r2 = curand_uniform(&state);
                float sqr1 = sqrtf(r1);
                float u = 1.0f - sqr1;
                float v = sqr1 * (1.0f - r2);
                // float w = sqr1 * r2; (1-u-v)
                Vec light_pos = light.v0 * u + light.v1 * v + light.v2 * (1.0f - u - v);

                // 3. 连线
                Vec to_light = light_pos - x_hit;
                float dist_sq = to_light.dot(to_light);
                if (dist_sq < 5) dist_sq = 5;
                float dist = sqrtf(dist_sq);
                Vec L_dir = to_light * (1.0f / dist);

                // 4. 几何检查
                float cos_theta = nl.dot(L_dir); // 表面是否朝向灯
                
                // 计算灯的法线
                Vec le1 = light.v1 - light.v0;
                Vec le2 = light.v2 - obj.v0; // 哎呀，这里要把 light.v0 写对
                // 其实可以直接在 Object 里存法线，或者现算
                Vec light_n = (light.v1 - light.v0).cross(light.v2 - light.v0).norm();
                float cos_light = -light_n.dot(L_dir); // 灯是否朝向表面

                if (cos_theta > 0.0f && cos_light > 0.0f) {
                    // 5. 阴影光线测试 (Shadow Ray)
                    // 注意：这里要传入 bvh_nodes 和 scene_objects
                    // 还要注意：trace_shadow 的 t_max 应该是 dist - 0.01 (稍微短一点，别打中灯自己)
                    if (!trace_shadow(x_hit + nl * 1e-3f, L_dir, dist - 1e-2f, bvh_nodes, scene_objects)) {
                        
                        // 6. 计算贡献
                        float area = triangle_area(light);
                        float pdf = 1.0f / (area * light_count); // 选这个点的概率密度
                        
                        // Geometry Term G = (cos_theta * cos_light) / dist^2
                        float G = (cos_theta * cos_light) / dist_sq;
                        
                        // BRDF (漫反射) = albedo / PI
                        Vec brdf = albedo * (1.0f / M_PI);
                        
                        // L_direct = Le * BRDF * G / PDF
                        Vec contribution = light.emission.mult(brdf) * (G / pdf);
                        
                        // 累加到 radiance
                        radiance = radiance + throughput.mult(contribution);
                    }
                }
            }
            // 漫反射颜色: 非金属才有
            Vec diffuse = albedo * (1.0f - metallic);

            // 随机采样半球
            float r1 = 2 * M_PI * curand_uniform(&state);
            float r2 = curand_uniform(&state);
            float r2s = sqrtf(r2);
            Vec w = nl;
            Vec temp = (fabs(w.x) > 0.1f ? make_vec(0, 1, 0) : make_vec(1, 0, 0));
            Vec u = temp.cross(w).norm();
            Vec v = w.cross(u);
            r_d = (u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2)).norm();
            
            // 能量更新
            // throughput *= diffuse / (1 - p_spec - p_trans)
            float p_diff = 1.0f - p_spec - (1.0f - p_spec) * transmission;
            float weight = 1.0f / p_diff;
            throughput = throughput.mult(diffuse) * weight; 

            r_o = x_hit + nl * 1e-3f;
            prev_refl_mode = DIFF; 
        }
        
    }

        // [修复] 终极防火墙：过滤 NaN 和 Inf
    // 只要 RGB 任何一个分量坏了，这一帧的采样就直接作废，不要污染历史数据。
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) ||
        isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
        return; // 直接丢弃这次采样
    }

    // [修复] 负值过滤
    // 某些极端情况下（如法线插值错误），可能算出负能量，这也会搞坏累加器
    if (radiance.x < 0.0f) radiance.x = 0.0f;
    if (radiance.y < 0.0f) radiance.y = 0.0f;
    if (radiance.z < 0.0f) radiance.z = 0.0f;

    // [修复] 亮度钳制 (Firefly Clamp)
    // 防止单帧超高亮度破坏平均值
    float max_lum = 100.0f; 
    float lum = radiance.x * 0.21 + radiance.y * 0.71 + radiance.z * 0.07;
    if (lum > max_lum) {
        radiance = radiance * (max_lum / lum);
    }
    
    // 写入显存
    accum_buffer[i] = accum_buffer[i] + radiance;
}

// Host 调用封装
// [修改 3] 封装函数负责传递全局指针 d_bvh_nodes
void launch_render_kernel(Vec* accum_buffer, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    dim3 threads(tx, ty);
    dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
    
    // 将全局变量 d_bvh_nodes 传进去
    render_kernel_impl<<<blocks, threads>>>(accum_buffer, width, height, frame_seed, cam, d_bvh_nodes, d_objects_ptr, d_light_indices, d_light_count);
}
