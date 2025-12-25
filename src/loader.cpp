#include "loader.h"
#include <cstdio>
#include <iostream>
#include <vector>

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
              Vec offset, float scale, Vec color, Refl_t refl, float fuzz) {
    
    // 打开文件 (只读模式)
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("[Loader Error] Cannot open file: %s\n", filename);
        return;
    }

    // [优化] 预留内存
    // 虽然我们不知道确切数量，但预留一点空间可以减少 vector 扩容时的内存拷贝开销。
    std::vector<Vec> temp_vertices;
    temp_vertices.reserve(2048); 

    char line[256]; // 缓冲区

    // 逐行读取
    while (fgets(line, sizeof(line), file)) {
        
        // ------------------------------------------------------------------
        // 解析顶点 (v x y z)
        // ------------------------------------------------------------------
        if (line[0] == 'v' && line[1] == ' ') {
            Vec v;
            // 解析 3 个浮点数
            sscanf(line, "v %f %f %f", &v.x, &v.y, &v.z);
            
            // [坐标变换]: 模型空间 -> 世界空间
            // 公式: v_world = v_model * scale + offset
            v = v * scale + offset;
            
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
            int matches = sscanf(line, "f %d %d %d", &idx[0], &idx[1], &idx[2]);
            
            // 只有成功解析出 3 个索引才处理
            if (matches == 3) {
                // [索引转换]: OBJ 是 1-based，C++ 数组是 0-based，所以要减 1
                // 安全检查: 防止索引越界 (虽然标准 OBJ 不会错，但防御性编程是好习惯)
                if (idx[0] < 1 || idx[0] > (int)temp_vertices.size() ||
                    idx[1] < 1 || idx[1] > (int)temp_vertices.size() ||
                    idx[2] < 1 || idx[2] > (int)temp_vertices.size()) {
                    continue; // 跳过非法面
                }

                Vec v0 = temp_vertices[idx[0] - 1];
                Vec v1 = temp_vertices[idx[1] - 1];
                Vec v2 = temp_vertices[idx[2] - 1];

                // [容量检查]
                // 防止物体数量超过 GPU 常量内存限制 (NUM_OBJECTS)
                if (objects.size() >= NUM_OBJECTS) {
                    printf("[Loader Warning] Scene full! Capped at %d objects.\n", NUM_OBJECTS);
                    break; // 停止加载后续三角形
                }

                // [构建三角形对象]
                // 使用 C++20 指定初始化器，清晰明了
                objects.push_back({
                    .v0 = v0, 
                    .v1 = v1, 
                    .v2 = v2,
                    .pos = {0,0,0}, // 三角形不需要中心点
                    .color = color,
                    .rad = 0,     // 三角形不需要半径
                    .tex_id = -1, // 暂不支持模型纹理
                    .fuzz = fuzz,
                    .refl = refl,
                    .type = TRIANGLE,
                });
            }
        }
    }

    fclose(file);
    
    // 打印统计信息
    printf("[Loader] Loaded: %s\n", filename);
    printf("         Vertices: %lu\n", temp_vertices.size());
    // objects.size() 可能包含之前的物体，所以这里不打印增加的数量了，只打印成功
}
