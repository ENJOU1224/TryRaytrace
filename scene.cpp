#include "scene.h"

Scene create_cornell_box() {
    Scene scene;
    // 使用 push_back 添加物体，这样代码更灵活，想加多少加多少
    scene.texture_files.push_back("earth.ppm");
    
    // 墙壁
    //                       rad/d      p(位置/法线)        e(自发光)     c(颜色)                     refl  type
    scene.objects.push_back({1.0f,      { 1, 0, 0},         {0,0,0},      {0.75f,   0.25f ,   0.25f },  DIFF, PLANE,  -1}); // 左
    scene.objects.push_back({-99.0f,    {-1, 0, 0},         {0,0,0},      {0.25f,   0.25f ,   0.75f },  DIFF, PLANE,  -1}); // 右
    scene.objects.push_back({0.0f,      { 0, 0, 1},         {0,0,0},      {0.75f,   0.75f ,   0.75f },  DIFF, PLANE,  0}); // 后
    scene.objects.push_back({-300.0f,   { 0, 0,-1},         {0,0,0},      {0.75f,   0.75f ,   0.75f },  DIFF, PLANE,  -1}); // 前
    scene.objects.push_back({0.0f,      { 0, 1, 0},         {0,0,0},      {0.75f,   0.75f ,   0.75f },  DIFF, PLANE,  -1}); // 地
    scene.objects.push_back({-81.6f,    { 0,-1, 0},         {0,0,0},      {0.75f,   0.75f ,   0.75f },  DIFF, PLANE,  -1}); // 顶
    
    // 球体
    scene.objects.push_back({16.5f,     {27, 16.5f, 47},    {0,0,0},      {0.999f,  0.999f, 0.999f},  SPEC, SPHERE, -1}); // 镜面
    scene.objects.push_back({16.5f,     {73, 16.5f, 78},    {0,0,0},      {0.999f,  0.999f, 0.999f},  REFR, SPHERE, -1}); // 玻璃
    
    // 灯光
    scene.objects.push_back({600.0f,    {50,681.33f,81.6f}, {12,12,12},   {0,0,0},                  DIFF, SPHERE, -1});
    
    return scene;
}
