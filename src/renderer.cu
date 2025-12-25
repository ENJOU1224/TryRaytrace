#include "renderer.h"
#include <curand_kernel.h> // CUDA 随机数生成库
#include <cstdio>          // 标准 IO (fopen, fscanf)
#include <cstring>         // 内存操作 (memset)
#include <string>

// ======================================================================================
// 全局显存数据 (Global GPU Memory)
// ======================================================================================

// [常量内存 (Constant Memory)]
// 存放场景中的物体数据。
// 特性:
// 1. 只有 64KB 大小，但对于我们的场景 (256个物体 * 96字节 = 24KB) 足够了。
// 2. 带有专用缓存 (Constant Cache)，当所有线程读取同一地址时速度极快 (广播机制)。
__constant__ Object d_objects[NUM_OBJECTS];

// [纹理句柄数组]
// 存放由 CUDA Runtime API 创建的纹理对象 (Texture Object)。
// 这是一个 64 位的整数句柄，指向显存中实际的纹理数据。
#define MAX_TEXTURES 5
__constant__ cudaTextureObject_t d_textures[MAX_TEXTURES]; 

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
void init_scene_data(const std::vector<Object>& objects, const std::vector<std::string>& texture_files) {
    // 1. 上传物体数据到常量内存
    int count = objects.size();
    if (count > NUM_OBJECTS) {
        printf("[Warning] Object count (%d) exceeds limit (%d). Truncating.\n", count, NUM_OBJECTS);
        count = NUM_OBJECTS;
    }
    cudaMemcpyToSymbol(d_objects, objects.data(), count * sizeof(Object));

    // 2. 加载并上传纹理
    std::vector<cudaTextureObject_t> temp_tex_objs;
    for (const auto& file : texture_files) {
        temp_tex_objs.push_back(load_texture_to_gpu(file));
    }
    
    // 3. 上传纹理句柄数组
    if (!temp_tex_objs.empty()) {
        cudaMemcpyToSymbol(d_textures, temp_tex_objs.data(), temp_tex_objs.size() * sizeof(cudaTextureObject_t));
    }
}

// ======================================================================================
// 设备端核心算法 (Device Kernel)
// ======================================================================================

// [函数]: 计算光线与物体的交点
// 返回值: t (距离)。如果未相交则返回 0.0。
__device__ float intersect(const Object& obj, const Vec& r_o, const Vec& r_d) {
    float eps = 1e-4f; // 防止自遮挡的微小偏移量

    // -----------------------------------------------------------
    // 三角形求交 (Möller–Trumbore 算法)
    // -----------------------------------------------------------
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
    return (t > eps) ? t : 0.0f;
}

// [内核]: 路径追踪主循环
__global__ void render_kernel_impl(Vec* accum_buffer, int width, int height, int frame_seed, CameraParams cam) {
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
    const int RR_THRESHOLD = 2;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // --- 场景遍历 (寻找最近交点) ---
        float t;
        int id = -1;
        float d_min = 1e20f;

        #pragma unroll
        for (int k = 0; k < NUM_OBJECTS; k++) {
            t = intersect(d_objects[k], r_o, r_d);
            if (t > 0 && t < d_min) { d_min = t; id = k; }
        }

        if (id < 0) break; // 没打中任何东西，结束

        const Object& obj = d_objects[id];
        Vec x_hit = r_o + r_d * d_min;
        
        // --- 法线计算 (几何属性) ---
        // [三角形法线]: 两条边做叉积
        // 注意: 这里我们使用"面法线" (Face Normal)，整个三角形是平的。
        // 如果要平滑渲染，需要重心坐标插值顶点法线 (Vertex Normal)，但那是进阶内容。
        Vec e1 = obj.v1 - obj.v0;
        Vec e2 = obj.v2 - obj.v0;
        Vec n = e1.cross(e2).norm();
        
        // 确保法线朝向观察者 (双面渲染)
        Vec nl = n.dot(r_d) < 0 ? n : n * -1;
        
        Vec f = obj.color;

        // --- 纹理采样 (材质属性) ---
        if (obj.tex_id >= 0) {
            float u = 0.0f, v = 0.0f;

            const float scale = 0.01f;
            // 根据法线朝向选择投影平面
            if (fabsf(n.y) > 0.9f)      { u = x_hit.x * scale; v = x_hit.z * scale; }
            else if (fabsf(n.x) > 0.9f) { u = x_hit.z * scale; v = x_hit.y * scale; }
            else                        { u = x_hit.x * scale; v = x_hit.y * scale; }
            v = 1.0f - v; // 翻转 V 轴适配纹理坐标系

            // 硬件采样
            float4 tex = tex2D<float4>(d_textures[obj.tex_id], u, v);
            f = f.mult(make_vec(tex.x, tex.y, tex.z));
        }

        // --- 自发光累加 ---
        radiance = radiance + throughput.mult(obj.emission);

        // --- 俄罗斯轮盘赌 (性能优化) ---
        if (depth > RR_THRESHOLD) {
            float p = f.x > f.y && f.x > f.z ? f.x : (f.y > f.z ? f.y : f.z);
            if (p < 1e-3f) p = 1e-3f;
            if (curand_uniform(&state) < p) f = f * (1.0f / p);
            else break;
        }
        
        throughput = throughput.mult(f);

        // --- BSDF 材质散射 (计算下一条光线) ---
        // 关键: 所有产生的 r_o 都要沿着法线方向推移 1e-3f，防止自我遮挡 (Shadow Acne)
        
        if (obj.refl == DIFF) {
            // [漫反射]: 余弦加权随机采样
            float r1 = 2 * M_PI * curand_uniform(&state);
            float r2 = curand_uniform(&state);
            float r2s = sqrtf(r2);
            Vec w = nl;
            Vec temp_axis = (fabs(w.x) > .1 ? make_vec(0, 1, 0) : make_vec(1, 0, 0));
            Vec u = temp_axis.cross(w).norm();
            Vec v = w.cross(u);
            r_d = (u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2)).norm();
            
            r_o = x_hit + nl * 1e-3f; 
        } 
        else if (obj.refl == SPEC) {
            // [镜面反射]: 完美反射
            Vec reflected = r_d - n * 2 * n.dot(r_d); 

            // 2. [新增] 粗糙度扰动
            // 在单位球内随机取一点
            // 简单且常用的方法：随机取单位圆盘上的点，甚至简单的立方体内取点归一化
            // 这里用 curand 生成随机单位向量
            float r1 = 2 * M_PI * curand_uniform(&state);
            float r2 = curand_uniform(&state);
            float z = 1.0f - 2.0f * r2;
            float r = sqrtf(1.0f - z * z);
            Vec random_sphere = make_vec(r * cosf(r1), r * sinf(r1), z);
            
            // 混合: 完美方向 + 粗糙度 * 随机方向
            // 然后归一化
            r_d = (reflected + random_sphere * obj.fuzz).norm();
            
            // 3. 吸收检查
            // 如果扰动太大，导致光线射向了物体内部 (和法线点积 < 0)，这物理上是不可能的(被挡住了)。
            // 这种情况下这束光线就"死"了 (被表面微结构吸收)。
            if (r_d.dot(n) <= 0.0f) {
                // 变成全黑，或者 break
                f = make_vec(0,0,0); // 吸收
                // 或者重新采样 (rejection sampling)，但这会分叉
                // 简单起见，吸收掉
                break;
            }
            r_o = x_hit + r_d * 1e-3f; // 沿着反射方向推
        } 
        else { // REFR
            // [折射]: 玻璃/水
            Vec refl = r_d - n * 2 * n.dot(r_d);
            bool into = n.dot(nl) > 0;
            float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc;
            float ddn = r_d.dot(nl);
            float cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
            
            if (cos2t < 0) { 
                // 全内反射 (Total Internal Reflection)
                r_d = refl; 
                r_o = x_hit + r_d * 1e-3f; 
            } else {
                // 菲涅尔分层 (Fresnel Split)
                Vec tdir = (r_d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)))).norm();
                float a = nt - nc, b = nt + nc, R0 = a * a / (b * b);
                float c = 1 - (into ? -ddn : tdir.dot(n));
                float Re = R0 + (1 - R0) * c * c * c * c * c;
                float Tr = 1 - Re;
                float P = .25f + .5f * Re, RP = Re / P, TP = Tr / (1 - P);
                
                if (curand_uniform(&state) < P) { 
                    throughput = throughput * RP; 
                    r_d = refl; 
                    r_o = x_hit + r_d * 1e-3f; 
                } else { 
                    throughput = throughput * TP; 
                    r_d = tdir; 
                    r_o = x_hit + r_d * 1e-3f; // 沿着折射方向推(穿过物体)
                }
            }
        }
    }
    
    // 写入显存
    accum_buffer[i] = accum_buffer[i] + radiance;
}

// Host 调用封装
void launch_render_kernel(Vec* accum_buffer, int width, int height, int frame_seed, int tx, int ty, CameraParams cam) {
    dim3 threads(tx, ty);
    dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
    render_kernel_impl<<<blocks, threads>>>(accum_buffer, width, height, frame_seed, cam);
}
