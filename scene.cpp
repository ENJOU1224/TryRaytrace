#include "scene.h"
#include <cstdio>
#include <vector>
#include <iostream>
#include "loader.h"

// ======================================================================================
// 场景创建
// ======================================================================================
Scene create_cornell_box() {
    Scene scene;
    scene.texture_files.push_back("earth.ppm");

    Vec white = {0.75f, 0.75f, 0.75f};
    Vec red   = {0.75f, 0.25f, 0.25f};
    Vec blue  = {0.25f, 0.25f, 0.75f};
    Vec gold  = {0.8f,  0.6f,  0.2f};
    Vec light = {12, 12, 12};

    // --- 1. 墙壁 ---
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=-1, .color=red,   .rad=1.0f,   .pos={ 1, 0, 0} }); // 左
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=-1, .color=blue,  .rad=-99.0f, .pos={-1, 0, 0} }); // 右
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=0,  .color={1,1,1},.rad=0.0f,   .pos={ 0, 0, 1} }); // 后(风景)
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=-1, .color={0,0,0},.rad=-300,   .pos={ 0, 0,-1} }); // 后(风景)
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=-1, .color=white, .rad=0.0f,   .pos={ 0, 1, 0} }); // 地
    scene.objects.push_back({ .refl=DIFF, .type=PLANE, .tex_id=-1, .color=white, .rad=-81.6f, .pos={ 0,-1, 0} }); // 顶

    // --- 2. 外部模型 (立方体) ---
    // 我们把 cube.obj 加载进来，放在空中，稍微旋转一点(这里没写旋转逻辑，只是放个位置)
    // 参数: 文件名, 目标数组, 位置偏移, 缩放大小, 颜色, 材质
    load_obj("cube.obj", scene.objects, 
             {50, 10,190},  // 位置 (左边空中)
             10.0f,         // 缩放 (变大10倍，因为原始坐标是1.0)
             gold,          // 金色
             SPEC);         // 镜面材质

    // --- 3. 球体 ---
    // 右下角的玻璃球保留
    scene.objects.push_back({ .refl=SPEC, .type=SPHERE, .tex_id=-1, .color={0.99f, 0.99f, 0.99f}, .rad=16.5f, .pos={27, 16.5f, 47} });
    scene.objects.push_back({ .refl=REFR, .type=SPHERE, .tex_id=-1, .color={0.99f, 0.99f, 0.99f}, .rad=16.5f, .pos={73, 16.5f, 78} });

    // --- 4. 灯光 ---
    scene.objects.push_back({ .refl=DIFF, .type=SPHERE, .tex_id=-1, .color={0,0,0}, .emission=light, .rad=600.0f, .pos={50, 681.33f, 81.6f} });

    return scene;
}
