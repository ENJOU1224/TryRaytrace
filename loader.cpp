#include "loader.h"
#include <cstdio>
#include <iostream>
#include <vector>

void load_obj(const char* filename, std::vector<Object>& objects, 
              Vec offset, float scale, Vec color, Refl_t refl) {
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open obj file %s\n", filename);
        return;
    }

    std::vector<Vec> temp_vertices;
    char line[128];

    while (fgets(line, sizeof(line), file)) {
        // 解析顶点 v x y z
        if (line[0] == 'v' && line[1] == ' ') {
            Vec v;
            sscanf(line, "v %f %f %f", &v.x, &v.y, &v.z);
            // 坐标变换
            v = v * scale + offset;
            temp_vertices.push_back(v);
        }
        // 解析面 f v1 v2 v3
        else if (line[0] == 'f' && line[1] == ' ') {
            int idx[3];
            int matches = sscanf(line, "f %d %d %d", &idx[0], &idx[1], &idx[2]);
            
            if (matches == 3) {
                // OBJ 索引从1开始，转为0开始
                Vec v0 = temp_vertices[idx[0] - 1];
                Vec v1 = temp_vertices[idx[1] - 1];
                Vec v2 = temp_vertices[idx[2] - 1];

                // 容量检查 (NUM_OBJECTS 定义在 scene.h 中)
                if (objects.size() >= NUM_OBJECTS) {
                    printf("Warning: Scene full! Capped at %d objects.\n", NUM_OBJECTS);
                    break;
                }

                // 构造三角形对象
                // 注意：使用 Designated Initializers (C++20) 写法最清晰
                objects.push_back({
                    .refl = refl,
                    .type = TRIANGLE,
                    .tex_id = -1,
                    .color = color,
                    .rad = 0, .pos = {0,0,0}, 
                    .v0 = v0, .v1 = v1, .v2 = v2
                });
            }
        }
    }

    fclose(file);
}
