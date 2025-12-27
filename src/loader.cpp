#include "loader.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// 简单的面结构，用于临时存储
struct FaceIdx { int v[3]; };

// ----------------------------------------------------------------------
// [辅助函数] 旋转向量 (Euler Angles: Pitch, Yaw, Roll)
// ----------------------------------------------------------------------
// p: 原始点
// r: 旋转角度 (x, y, z) 单位: 度
Vec rotate_point(Vec p, Vec r) {
    // 转弧度
    float rad_x = r.x * (M_PI / 180.0f);
    float rad_y = r.y * (M_PI / 180.0f);
    float rad_z = r.z * (M_PI / 180.0f);

    // 1. 绕 X 轴旋转
    if (r.x != 0.0f) {
        float y = p.y * cosf(rad_x) - p.z * sinf(rad_x);
        float z = p.y * sinf(rad_x) + p.z * cosf(rad_x);
        p.y = y; p.z = z;
    }

    // 2. 绕 Y 轴旋转
    if (r.y != 0.0f) {
        float x = p.x * cosf(rad_y) + p.z * sinf(rad_y);
        float z = -p.x * sinf(rad_y) + p.z * cosf(rad_y);
        p.x = x; p.z = z;
    }

    // 3. 绕 Z 轴旋转
    if (r.z != 0.0f) {
        float x = p.x * cosf(rad_z) - p.y * sinf(rad_z);
        float y = p.x * sinf(rad_z) + p.y * cosf(rad_z);
        p.x = x; p.y = y;
    }

    return p;
}

// ======================================================================================
// Wavefront OBJ 模型加载器
// ======================================================================================
// [文件格式简介]
// OBJ 是一种基于文本的 3D 模型格式。
// 1. 顶点 (Vertex): 以 'v' 开头，后面跟着 x y z 坐标。
//    v 1.0 -1.0 0.0
// 2. 面 (Face): 以 'f' 开头，后面跟着顶点的索引 (从1开始计数)。
//    f 1 2 3
//
// [CPU 预处理]
// 我们在加载时直接对顶点进行了缩放 (Scale) 和平移 (Offset)。
// 这样 GPU 拿到的就是已经是世界坐标系下的三角形，无需在 Kernel 里再做矩阵变换，
// 省去了 GPU 的计算压力 (Baking transform)。
// ======================================================================================
void load_obj(const char* filename, std::vector<Object>& objects, 
              Vec offset, float scale, Vec rotation, // [新增]
              Vec albedo, float metallic, float roughness, float transmission, float ior,
              int tex_id) {
    
    // 打开文件 (只读模式)
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("[Loader Error] Cannot open file: %s\n", filename);
        return;
    }

    // [优化] 预留内存
    // 虽然我们不知道确切数量，但预留一点空间可以减少 vector 扩容时的内存拷贝开销。
    std::vector<Vec> temp_vertices;
    std::vector<FaceIdx> temp_faces;
    temp_vertices.reserve(2048); 

    char line[256]; // 缓冲区

    // 1. 第一遍扫描: 读取顶点和面索引
    while (fgets(line, sizeof(line), file)) {
        
        // ------------------------------------------------------------------
        // 解析顶点 (v x y z)
        // ------------------------------------------------------------------
        if (line[0] == 'v' && line[1] == ' ') {
            Vec v;
            // 解析 3 个浮点数
            sscanf(line, "v %f %f %f", &v.x, &v.y, &v.z);
            
            // [坐标变换]: 模型空间 -> 世界空间
            // [变换管线]: 缩放 -> 旋转 -> 平移
            v = v * scale;           // 1. 缩放
            v = rotate_point(v, rotation); // 2. 旋转 [新增]
            v = v + offset;          // 3. 平移
            
            temp_vertices.push_back(v);
        }
        
        // ------------------------------------------------------------------
        // 解析面 (f v1 v2 v3)
        // ------------------------------------------------------------------
        // 注意: OBJ 格式可能很复杂 (如 f 1/1/1 2/2/2)。
        // 这里的代码假设你导出时只勾选了 "Triangulate Faces" 且没有勾选 "Write Normals/UVs"。
        // 格式必须是简单的 "f 1 2 3"。
        else if (line[0] == 'f' && line[1] == ' ') {
            int idx[3];

            // 尝试解析 f v1 v2 v3 (最简格式)
            // 这里的 %*s 是跳过可能的 /vt/vn 部分
            int matches = sscanf(line, "f %d%*s %d%*s %d%*s", &idx[0], &idx[1], &idx[2]);
            
            // 如果解析失败，尝试纯数字格式
            if (matches != 3) {
                 matches = sscanf(line, "f %d %d %d", &idx[0], &idx[1], &idx[2]);
            }
            
            // 只有成功解析出 3 个索引才处理
            if (matches == 3) {
                // [索引转换]: OBJ 是 1-based，C++ 数组是 0-based，所以要减 1
                // 安全检查: 防止索引越界 (虽然标准 OBJ 不会错，但防御性编程是好习惯)
                if (idx[0] < 1 || idx[0] > (int)temp_vertices.size() ||
                    idx[1] < 1 || idx[1] > (int)temp_vertices.size() ||
                    idx[2] < 1 || idx[2] > (int)temp_vertices.size()) {
                    continue; // 跳过非法面
                }
                temp_faces.push_back({idx[0]-1, idx[1]-1, idx[2]-1});
            }
        }
    }

    fclose(file);
    
    // 2. 自动计算平滑法线 (Vertex Normal Averaging)
    // 因为你的 OBJ 没有 vn 数据，我们需要自己算
    printf("[Loader] Computing smooth normals for %lu faces...\n", temp_faces.size());

    // 创建一个累加器，大小等于顶点数，初始化为 0
    std::vector<Vec> vertex_normals(temp_vertices.size(), {0,0,0});

    // 遍历所有面，计算面法线，并累加到对应的顶点上
    for (const auto& face : temp_faces) {
        // 安全检查
        if (face.v[0] >= temp_vertices.size() || 
            face.v[1] >= temp_vertices.size() || 
            face.v[2] >= temp_vertices.size()) continue;

        Vec p0 = temp_vertices[face.v[0]];
        Vec p1 = temp_vertices[face.v[1]];
        Vec p2 = temp_vertices[face.v[2]];

        // 面法线 = (p1-p0) X (p2-p0)
        // 注意：这里不归一化。
        // 不归一化的好处是：面积大的三角形会对顶点法线产生更大的权重贡献 (Area Weighted)，
        // 这通常比简单的平均效果更好。
        Vec e1 = p1 - p0;
        Vec e2 = p2 - p0;
        Vec face_n = e1.cross(e2); 

        // 累加到三个顶点
        vertex_normals[face.v[0]] = vertex_normals[face.v[0]] + face_n;
        vertex_normals[face.v[1]] = vertex_normals[face.v[1]] + face_n;
        vertex_normals[face.v[2]] = vertex_normals[face.v[2]] + face_n;
    }

    // 归一化所有顶点法线
    for (auto& n : vertex_normals) {
        n.norm();
    }

    // 3. 组装 Object 数据
    // 将顶点、计算好的法线、材质参数打包进 Object
    for (const auto& face : temp_faces) {
        // 显存容量检查 (现在是 Global Memory，通常不会溢出，但检查一下是个好习惯)
        // 假设我们不想无限加载...
        // if (objects.size() >= MAX_SCENE_OBJECTS) ... (现在可以放宽了)

        Object obj;
        
        // 几何数据
        obj.v0 = temp_vertices[face.v[0]];
        obj.v1 = temp_vertices[face.v[1]];
        obj.v2 = temp_vertices[face.v[2]];

        // 法线数据 (使用计算好的平滑法线)
        obj.vn0 = vertex_normals[face.v[0]];
        obj.vn1 = vertex_normals[face.v[1]];
        obj.vn2 = vertex_normals[face.v[2]];
        
        // 默认 UV (因为文件里没有)
        // 简单的重心坐标占位，防止未初始化
        obj.uv0 = {0.0f, 0.0f}; 
        obj.uv1 = {1.0f, 0.0f}; 
        obj.uv2 = {0.0f, 1.0f};

        // 材质数据
        obj.albedo = albedo;
        obj.emission = {0,0,0}; // 模型默认不发光
        obj.metallic = metallic;
        obj.roughness = roughness;
        obj.transmission = transmission;
        obj.ior = ior;
        obj.tex_id = tex_id;
        
        // 标记: 使用平滑插值
        obj.use_smooth = 1; 

        objects.push_back(obj);
    }

    printf("[Loader] Loaded: %s (%lu triangles)\n", filename, objects.size());
}


// [辅助函数] 手写 P6 图片加载器 (零依赖)
// 解析 PPM (P6 Binary) 格式图片
// 返回: 原始 RGB 数据的堆内存指针 (调用者需负责 free)
unsigned char* load_ppm(const char* filename, int* w, int* h) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "[Texture Error] Cannot open file: %s\n", filename);
        return nullptr;
    }

    char header[64];
    // 读取文件头，忽略 fscanf 的返回值警告 (但在生产环境中应检查)
    if (fscanf(fp, "%63s", header) != 1) {} 

    if (strcmp(header, "P6") != 0) {
        fprintf(stderr, "[Texture Error] Not a P6 binary PPM: %s\n", filename);
        fclose(fp);
        return nullptr;
    }

    int max_val;
    if (fscanf(fp, "%d %d %d", w, h, &max_val) != 3) {}
    
    // [细节]: 吃掉 header 后的换行符，否则会错读像素数据
    fgetc(fp); 

    // 分配内存 (RGB 3通道)
    size_t bytes = (size_t)(*w) * (*h) * 3;
    unsigned char* data = (unsigned char*)malloc(bytes);
    
    // 读取二进制像素块
    if (fread(data, 1, bytes, fp) != bytes) {
        fprintf(stderr, "[Texture Error] Unexpected EOF: %s\n", filename);
        free(data);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    printf("[Texture] Loaded: %s (%dx%d)\n", filename, *w, *h);
    return data;
}

