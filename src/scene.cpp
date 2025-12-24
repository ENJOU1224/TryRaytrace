#include "scene.h"
#include "loader.h" // 引入 OBJ 加载器

// ======================================================================================
// 工厂函数: create_cornell_box()
// ======================================================================================
// 职责:
// 1. 定义场景中的几何体 (墙壁、球体、模型)。
// 2. 指定每个物体的材质和纹理。
// 3. 将所有物体打包成一个 Scene 对象返回给 Main。
//
// 架构角色: "数据提供者 (Data Provider)"
// ======================================================================================
Scene create_cornell_box() {
    Scene scene;

    // ------------------------------------------------------------------
    // 1. 资源注册
    // ------------------------------------------------------------------
    // 注册所有需要加载的纹理文件。
    // ID 0: "assets/earth.ppm"
    scene.texture_files.push_back("assets/earth.ppm");

    // ------------------------------------------------------------------
    // 2. 材质与颜色预定义
    // ------------------------------------------------------------------
    // 预定义颜色可以提高代码可读性，避免魔法数字
    Vec white = {0.75f, 0.75f, 0.75f};
    Vec red   = {0.75f, 0.25f, 0.25f};
    Vec blue  = {0.25f, 0.25f, 0.75f};
    Vec gold  = {0.8f,  0.6f,  0.2f};
    Vec light = {12.0f, 12.0f, 12.0f}; // 高强度自发光

    // ------------------------------------------------------------------
    // 3. 物体构建 (Procedural Scene Generation)
    // ------------------------------------------------------------------
    // 使用 C++20 指定初始化器 (.field = value) 语法，清晰且不易出错。
    
    // --- 墙壁 (Plane) ---
    // [坐标系]: X(左右), Y(上下), Z(前后)
    // [平面方程]: N·P = D,  其中 N = pos, D = rad
    
    scene.objects.push_back({ .pos={ 1, 0, 0}, .color=red    ,  .rad=1.0f   , .tex_id=-1, .refl=DIFF, .type=PLANE});    // 左墙: x=1,   法线朝右
    scene.objects.push_back({ .pos={-1, 0, 0}, .color=blue   ,  .rad=-99.0f , .tex_id=-1, .refl=DIFF, .type=PLANE});    // 右墙: x=99,  法线朝左
    scene.objects.push_back({ .pos={ 0, 0, 1}, .color={1,1,1},  .rad=0.0f   , .tex_id=0 , .refl=DIFF, .type=PLANE});    // 后墙: z=0,   法线朝前 (贴风景图)
    scene.objects.push_back({ .pos={ 0, 1, 0}, .color=white  ,  .rad=0.0f   , .tex_id=-1, .refl=DIFF, .type=PLANE});    // 地板: y=0,   法线朝上
    scene.objects.push_back({ .pos={ 0,-1, 0}, .color=white  ,  .rad=-81.6f , .tex_id=-1, .refl=DIFF, .type=PLANE});    // 天花板: y=81.6, 法线朝下
    // 注意: 前墙已移除，允许相机自由进出

    // --- 外部模型 (Mesh) ---
    // 调用 loader 模块来加载 cube.obj 文件
    // 参数: 文件名, 目标容器, 位置偏移, 缩放大小, 颜色, 材质
    load_obj("assets/cube.obj", scene.objects, 
             {50.0f, 10.0f, 130.0f}, // 位置: 放在盒子中间偏上
             10.0f,                 // 缩放: 模型原始大小是 -1到1，放大10倍
             gold,                  // 颜色: 金色
             SPEC);                 // 材质: 镜面

    // --- 球体 (Sphere) ---
    scene.objects.push_back({ 
        .pos={73.0f, 16.5f, 78.0f},
        .color={0.99f, 0.99f, 0.99f}, 
        .rad=16.5f, 
        .tex_id=-1, 
        .refl=REFR, 
        .type=SPHERE
    }); // 玻璃球 (右侧)
    scene.objects.push_back({ 
        .pos={27.0f, 16.5f, 47.0f},
        .color={0.99f, 0.99f, 0.99f}, 
        .rad=16.5f, 
        .tex_id=-1, 
        .refl=SPEC, 
        .type=SPHERE
    }); // 玻璃球 (右侧)

    // --- 光源 (Light) ---
    // 这是一个自发光的球体，模拟吸顶灯
    scene.objects.push_back({ 
        .pos={50.0f, 681.33f, 81.6f},
        .color={0,0,0},         // 灯的表面颜色是黑的 (不反射)
        .emission=light,        // 但它自己会发光
        .rad=600.0f, 
        .tex_id=-1, 
        .refl=DIFF,       // 灯本身是漫反射材质
        .type=SPHERE, 
    });

    // 打印场景信息
    printf("[Scene] Scene created with %lu objects.\n", scene.objects.size());

    return scene;
}
