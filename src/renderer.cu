#include "renderer.h"
#include <curand_kernel.h> // CUDA 的随机数生成库 (这是 GPU 上的硬件随机数生成器的软件接口)
#include <stdio.h>
#include <string>

// ======================================================================================
// 全局显存数据 (Global/Constant Memory)
// ======================================================================================

// [CUDA 硬件特性]: __constant__ 常量内存
// 这是一个位于显存(VRAM)中的特殊区域，大小通常限制在 64KB。
// 优势:
// 1. 拥有专属的 L1 Cache (Constant Cache)。
// 2. 广播机制: 当 Warp 中的所有线程读取同一个地址时(比如大家都在读第0个物体)，
//    只需要一次显存事务，数据会"广播"给这 32 个线程。这比普通 Global Memory 快得多。
// 适用场景: 场景数据、配置参数等只读且所有线程都会频繁访问的数据。
__constant__ Object d_objects[NUM_OBJECTS];

// ============================================================================
// 1. 纹理资源管理
// ============================================================================

// 最多支持 5 张纹理
#define MAX_TEXTURES 5
// 纹理对象句柄数组 (存在常量内存中)
__constant__ cudaTextureObject_t d_textures[MAX_TEXTURES]; 

// [纯手写 PPM (P6) 读取器]
// 返回: 像素数据指针 (RGB, 3字节/像素)
// 输出参数: w, h
// renderer.cu 里的 load_ppm 函数
unsigned char* load_ppm(const char* filename, int* w, int* h) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open texture file %s\n", filename);
        return NULL;
    }

    char header[64];
    if (fscanf(fp, "%s", header) != 1) { /* handle error */ }
    // 1. 读取 P6 头
    if (strcmp(header, "P6") != 0) {
        printf("Error: Not a P6 binary PPM file.\n");
        fclose(fp);
        return NULL;
    }

    // 2. 读取宽、高、最大值
    int max_val;
    // [修正] 检查返回值
    if (fscanf(fp, "%d %d %d", w, h, &max_val) != 3) { /* handle error */ }
    
    // [关键细节]: fscanf 读完数字后，文件指针停在换行符前。
    // 我们必须吃掉这个换行符，否则它会被当成像素数据的第一个字节。
    fgetc(fp); 

    // 3. 分配内存
    // PPM 是 RGB (3通道)
    size_t bytes = (*w) * (*h) * 3;
    unsigned char* data = (unsigned char*)malloc(bytes);

    // [修正] 检查返回值
    if (fread(data, 1, bytes, fp) != bytes) {
        printf("Error: Unexpected EOF in texture file.\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    // 4. 读取二进制像素数据
    fclose(fp);

    printf("Texture loaded: %s (%dx%d)\n", filename, *w, *h);
    return data;
}

// renderer.cu

cudaTextureObject_t load_texture_to_gpu(const std::string& filename) {
    int w, h;
    
    // 1. 调用我们可以手写的 Loader (RGB 数据)
    unsigned char* rgb_data = load_ppm(filename.c_str(), &w, &h);
    if (!rgb_data) return 0;

    // 2. [格式转换] RGB (3 bytes) -> RGBA (4 bytes)
    // 为了满足 CUDA 对齐要求，我们手动扩展数据
    unsigned char* rgba_data = (unsigned char*)malloc(w * h * 4);
    for (int i = 0; i < w * h; i++) {
        rgba_data[i*4 + 0] = rgb_data[i*3 + 0]; // R
        rgba_data[i*4 + 1] = rgb_data[i*3 + 1]; // G
        rgba_data[i*4 + 2] = rgb_data[i*3 + 2]; // B
        rgba_data[i*4 + 3] = 255;               // A (不透明)
    }
    
    // 用完 RGB 数据就可以扔了
    free(rgb_data);

    // 3. 分配 CUDA Array (指定 uchar4 格式)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, w, h);

    // 4. 拷贝 RGBA 数据 CPU -> GPU
    const size_t spitch = w * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(cuArray, 0, 0, rgba_data, spitch, w * 4 * sizeof(unsigned char), h, cudaMemcpyHostToDevice);

    // 清理临时 RGBA 数据
    free(rgba_data);

    // 5. 创建资源描述符 (保持不变)
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 6. 创建纹理描述符 (保持不变)
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear; // 硬件双线性插值
    texDesc.readMode = cudaReadModeNormalizedFloat; // 归一化到 [0,1]
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}


// ======================================================================================
// 宿主端接口实现 (Host Implementation)
// ======================================================================================

// 这个函数运行在 CPU 上，负责把数据搬给 GPU
void init_scene_data(const std::vector<Object>& objects, const std::vector<std::string>& texture_files) {
    // 上传物体
    int count = objects.size();
    if (count > NUM_OBJECTS) count = NUM_OBJECTS;
    cudaMemcpyToSymbol(d_objects, objects.data(), count * sizeof(Object));

    // 上传纹理
    std::vector<cudaTextureObject_t> temp_tex_objs;
    for (const auto& file : texture_files) {
        temp_tex_objs.push_back(load_texture_to_gpu(file));
    }
    
    // 把句柄列表传给 GPU
    if (!temp_tex_objs.empty()) {
        cudaMemcpyToSymbol(d_textures, temp_tex_objs.data(), temp_tex_objs.size() * sizeof(cudaTextureObject_t));
    }
}

// ======================================================================================
// 设备端函数 (Device Functions) - 只有 GPU 能调用
// ======================================================================================

// [物理核心]: 射线求交
// 这是一个纯数学函数，没有显存读写，全是 ALU 运算。
// 现代 GPU (如 Turing) 有两套 ALU (FP32 和 INT32)，这里主要榨干 FP32 单元。
__device__ float intersect(const Object& obj, const Vec& r_o, const Vec& r_d) {
    float eps = 1e-4f; // 极小值，用于消除浮点误差带来的"自我遮挡"
    
    // 分支 1: 平面求交
    if (obj.type == TRIANGLE) {
        // [Möller–Trumbore 算法]
        // 1. 计算两边向量 (可以在初始化时算好存起来，这里为了简单现场算)
        Vec e1 = obj.v1 - obj.v0;
        Vec e2 = obj.v2 - obj.v0;
        
        // 2. 计算行列式
        Vec h = r_d.cross(e2);
        float a = e1.dot(h);
        
        // 如果 a 接近 0，说明光线平行于三角形，没打中
        if (a > -eps && a < eps) return 0.0f;
        
        float f = 1.0f / a;
        Vec s = r_o - obj.v0;
        float u = f * s.dot(h);
        
        // 检查重心坐标 u 范围
        if (u < 0.0f || u > 1.0f) return 0.0f;
        
        Vec q = s.cross(e1);
        float v = f * r_d.dot(q);
        
        // 检查重心坐标 v 范围
        if (v < 0.0f || u + v > 1.0f) return 0.0f;
        
        // 3. 计算 t (距离)
        float t = f * e2.dot(q);
        
        if (t > eps) return t;
        else return 0.0f;
    }else if (obj.type == PLANE) {
        // 公式: t = (D - N·O) / (N·D)
        float denom = obj.pos.dot(r_d);
        // fabsf: 浮点绝对值。如果分母太小，说明光线平行于平面，打不中。
        if (fabsf(denom) > 1e-6f) {
            float t = (obj.rad - obj.pos.dot(r_o)) / denom;
            return (t > eps) ? t : 0.0f; // 只返回正前方的交点
        }
        return 0.0f;
    } 
    // 分支 2: 球体求交
    else {
        // 求解一元二次方程: at^2 + bt + c = 0
        Vec op = obj.pos - r_o;
        float b = op.dot(r_d);
        float det = b * b - op.dot(op) + obj.rad * obj.rad;
        
        if (det < 0) return 0; else det = sqrtf(det); // 判别式<0，无解
        
        float t;
        // 先试算较近的交点 (进球点)
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
}

// ======================================================================================
// 渲染内核 (Render Kernel) - GPU 入口点
// ======================================================================================
// __global__: 表示这是 CPU 调用的入口，GPU 执行。
__global__ void render_kernel_impl(Vec* accum_buffer, int width, int height, int frame_seed, CameraParams cam) {
    
    // ------------------------------------------------------------------
    // 1. 线程索引计算 (硬件映射)
    // ------------------------------------------------------------------
    // x, y 是当前线程负责的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：防止溢出
    if (x >= width || y >= height) return;

    // [关键坐标变换]: Y 轴翻转
    // 物理/数学世界中，Y=0 通常在地板。
    // 屏幕/存储世界中，索引 0 通常在左上角。
    // 这里我们把 (x, y) 映射到 1D 数组时，让 y=0 对应数组末尾(屏幕底部)，y=max 对应数组开头(屏幕顶部)。
    // 这样算出来的图就是正的。
    int i = (height - y - 1) * width + x;

    // ------------------------------------------------------------------
    // 2. 随机数状态初始化
    // ------------------------------------------------------------------
    // 每个像素需要独立的随机序列。
    // 1984 是基准种子，frame_seed 是帧偏移。
    // 这样每一帧、每个像素生成的随机数都不一样，保证蒙特卡洛积分有效。
    curandState state;
    curand_init(1984 + frame_seed, i, 0, &state);

    // ------------------------------------------------------------------
    // 3. 抗锯齿采样 (Sub-pixel Jitter)
    // ------------------------------------------------------------------
    // 不射向像素中心，而是在像素范围内抖动。
    // curand_uniform 返回 (0.0, 1.0] 的均匀分布
    float r1 = 2 * curand_uniform(&state);
    float r2 = 2 * curand_uniform(&state);
    // Tent Filter 算法: 让采样更集中在像素中心，边缘较少
    float dx = r1 < 1 ? sqrtf(r1) - 1 : 1 - sqrtf(2 - r1);
    float dy = r2 < 1 ? sqrtf(r2) - 1 : 1 - sqrtf(2 - r2);
    
    // 生成光线方向
    Vec dir_pinhole = (cam.cx * (((x + .5f + dx) / width - .5f)) + 
                       cam.cy * (((y + .5f + dy) / height - .5f)) + cam.dir).norm();
    
    // 2. 采样透镜上的点 (Lens Sampling)
    Vec lens_offset = {0, 0, 0};
    if (cam.lens_radius > 0.0f) {
        // 在单位圆内随机取点 (极坐标法)
        float lr = cam.lens_radius * sqrtf(curand_uniform(&state));
        float ltheta = 2 * M_PI * curand_uniform(&state);
        
        // 我们需要透镜平面的基向量 (u, v)。
        // 简单做法：直接归一化 cx 和 cy，因为它们必定垂直于 dir。
        Vec u = cam.cx.norm();
        Vec v = cam.cy.norm();
        
        // 计算偏移量
        lens_offset = u * (lr * cosf(ltheta)) + v * (lr * sinf(ltheta));
    }

    // 3. 计算光线
    // [焦点]: 从相机中心出发，沿着针孔方向走 focus_dist 远
    Vec p_focus = cam.pos + dir_pinhole * cam.focus_dist;
    
    // [起点]: 相机中心 + 透镜偏移
    Vec r_o = cam.pos + lens_offset;
    
    // [方向]: 从新的起点指向焦点
    Vec r_d = (p_focus - r_o).norm();

    // ------------------------------------------------------------------
    // 4. 路径追踪循环 (Path Tracing Loop)
    // ------------------------------------------------------------------
    Vec throughput = make_vec(1, 1, 1); // 能量传输比率 (颜色滤镜)
    Vec radiance = make_vec(0, 0, 0);   // 收集到的光能
                                        //
    // [配置]
    const int MAX_DEPTH = 10;      // 硬上限: 保证 GPU 帧率稳定 (最坏情况只算8次)
    const int RR_THRESHOLD = 2;   // 软上限: 超过这个深度开始赌运气

    int depth = 0;

    // 循环 8 次 (光线最多反弹 8 次)
    // 经验值: 8 次足够照亮玻璃球内部和天花板阴影
    for (depth = 0; depth < MAX_DEPTH; depth++) {
        float t;
        int id = -1;
        float d_min = 1e20f;

        // [编译器优化]: 循环展开
        // 告诉编译器这里的循环次数很少且固定，直接展开成线性指令，
        // 消除循环计数器跳转 (Branch) 的开销。
        #pragma unroll
        for (int k = 0; k < NUM_OBJECTS; k++) {
            t = intersect(d_objects[k], r_o, r_d);
            if (t > 0 && t < d_min) { d_min = t; id = k; }
        }

        // 没打中任何东西，光线飞向虚空，结束
        if (id < 0) break;

        const Object& obj = d_objects[id];
        Vec x_hit = r_o + r_d * d_min; // 交点
        Vec n;
        if (obj.type == SPHERE) {
            n = (x_hit - obj.pos).norm();
        } else if (obj.type == PLANE) {
            n = obj.pos; // 这里的 pos 存的是法线
        } else if (obj.type == TRIANGLE) {
            // 三角形法线 = 两边叉积
            Vec e1 = obj.v1 - obj.v0;
            Vec e2 = obj.v2 - obj.v0;
            n = e1.cross(e2).norm();
        }
        Vec nl = n.dot(r_d) < 0 ? n : n * -1; // 确保法线朝向光线来的一侧
        Vec f = obj.color; // 物体颜色
                       
        // [新增] 纹理采样逻辑 (UV Mapping)
        if (obj.tex_id >= 0) {
            float u = 0.0f, v = 0.0f;
            
            // --- 情况 A: 球体 (地球仪) ---
            if (obj.type == SPHERE) {
                Vec p = (x_hit - obj.pos).norm(); 
                float phi = atan2f(p.z, p.x);
                float theta = asinf(p.y);
                u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
                v = (theta + M_PI / 2.0f) / M_PI;
            } 
            
            // --- 情况 B: 平面 (墙壁/风景画) [新增部分] ---
            else if (obj.type == PLANE) {
                // 平面映射原理：
                // 墙是平的，我们直接把交点的 (x, y, z) 坐标映射到 (u, v)。
                // Cornell Box 的尺寸大约是 0 到 100。
                // 纹理坐标 UV 需要 0.0 到 1.0。
                // 所以我们需要除以一个缩放系数 (比如 100.0)。
                
                const float scale = 0.01f; // 1.0 / 100.0
                
                // 我们根据墙的朝向(法线 n)来决定用哪两个坐标轴
                // fabsf 是取绝对值
                if (fabsf(n.y) > 0.9f) {
                    // 1. 地板/天花板 (法线朝 Y): 映射 X, Z
                    u = x_hit.x * scale;
                    v = x_hit.z * scale;
                } 
                else if (fabsf(n.x) > 0.9f) {
                    // 2. 左右墙 (法线朝 X): 映射 Z, Y
                    u = x_hit.z * scale;
                    v = x_hit.y * scale;
                } 
                else {
                    // 3. 前后墙 (法线朝 Z): 映射 X, Y  <-- 你的风景图会走这里
                    u = x_hit.x * scale;
                    v = x_hit.y * scale;
                }
                v = 1.0f - v; 
                // [可选] 如果发现图片倒了，可以用 v = 1.0f - v;
            }

            // [硬件加速采样]
            // 使用 wrap 模式（d_textures设置里是 cudaAddressModeWrap）
            // 这样如果墙很大，图片会自动重复平铺
            float4 tex_color = tex2D<float4>(d_textures[obj.tex_id], u, v);
            
            // 混合颜色
            // 注意：如果你想让风景图完全显示，把墙壁的基础色 c 改成白色 {1,1,1}
            // 否则风景图会混上墙原本的颜色 (比如混上黑色就看不见了)
            f = f.mult(make_vec(tex_color.x, tex_color.y, tex_color.z));
        }

        // 累加自发光 (如果是灯，e > 0; 如果是墙，e = 0)
        radiance = radiance + throughput.mult(obj.emission);
        
        // [俄罗斯轮盘赌]: 概率性终止光线
        // 目的: 减少深层递归的计算量。如果 throughput 已经很黑了，没必要算了。
        // 阈值: 5。前 5 次必算，后面看概率。
        if (depth > RR_THRESHOLD) {
            // 计算最大反射率作为存活概率
            float p = f.x > f.y && f.x > f.z ? f.x : (f.y > f.z ? f.y : f.z);
            
            // 防止概率太小导致除以零 (虽然在场景里不太可能)
            if (p < 1e-3f) p = 1e-3f;

            if (curand_uniform(&state) < p) f = f * (1 / p); // 存活，能量补偿
            else break; // 死亡
        }
        
        // 更新能量传输率
        throughput = throughput.mult(f);

        // --- 材质交互 (BSDF) ---
        
        if (obj.refl == DIFF) {
            // [漫反射]: 随机半球采样
            float r1 = 2 * M_PI * curand_uniform(&state);
            float r2 = curand_uniform(&state);
            float r2s = sqrtf(r2);
            Vec w = nl;
            Vec temp_axis = (fabs(w.x) > .1 ? make_vec(0, 1, 0) : make_vec(1, 0, 0));
            Vec u = temp_axis.cross(w).norm();
            Vec v = w.cross(u);
            r_d = (u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2)).norm();
            r_o = x_hit;
        } 
        else if (obj.refl == SPEC) {
            // [镜面反射]: 完美反射
            r_d = r_d - n * 2 * n.dot(r_d); 
            // Ray Epsilon: 把起点往外推一点，防止光线刚出门就打中自己 (Self-Intersection)
            r_o = x_hit + n * 1e-3f; 
        } 
        else {
            // [折射]: 玻璃
            Vec refl = r_d - n * 2 * n.dot(r_d);
            bool into = n.dot(nl) > 0;
            float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc;
            float ddn = r_d.dot(nl);
            float cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
            
            if (cos2t < 0) { 
                // 全内反射
                r_d = refl; 
                r_o = x_hit + n * 1e-3f; 
            } else {
                // 菲涅尔效应 (Fresnel Effect)
                Vec tdir = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)))).norm();
                float a = nt - nc, b = nt + nc, R0 = a * a / (b * b);
                float c = 1 - (into ? -ddn : tdir.dot(n));
                float Re = R0 + (1 - R0) * c * c * c * c * c;
                float Tr = 1 - Re;
                float P = .25f + .5f * Re, RP = Re / P, TP = Tr / (1 - P);
                
                if (curand_uniform(&state) < P) { 
                    throughput = throughput * RP; 
                    r_d = refl; 
                    r_o = x_hit + n * 1e-3f; // 反射往外推
                } else { 
                    throughput = throughput * TP; 
                    r_d = tdir; 
                    r_o = x_hit + r_d * 1e-3f; // 折射往里推 (透过去)
                }
            }
        }
    }
    
    // [写入显存]: 累积混合
    // 多个线程可能同时在写 accum_buffer 的不同位置，但因为 i 是唯一的，
    // 所以这里不需要 atomicAdd (除非我们要多个线程算同一个像素)。
    accum_buffer[i] = accum_buffer[i] + radiance;
}

// ======================================================================================
// 宿主端包装函数 (Wrapper)
// ======================================================================================
// 这个函数是普通的 C++ 函数，可以在 main.cpp 里调用。
// 它负责计算 Grid/Block 维度，并启动 Kernel。
void launch_render_kernel(Vec* accum_buffer, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    // 设定 Block 大小 (例如 16x16)
    dim3 threads(tx, ty);
    // 设定 Grid 大小 (向上取整，覆盖整个图像)
    dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
    
    // 发射!
    render_kernel_impl<<<blocks, threads>>>(accum_buffer, width, height, frame_seed, cam);
    
    // 这里的 Kernel 启动是异步的，CPU 发完指令立刻返回，不等待 GPU 结束。
}
