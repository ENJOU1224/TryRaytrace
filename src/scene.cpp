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

    // --- 墙壁 (Walls) - 用三角形替代平面 ---
    // [坐标系]: X(左右), Y(上下), Z(前后)
    // [范围]: -100 到 100
    // [注意]: 前墙已移除，允许相机自由进出

    // 左墙 (x=1, 红色, 法线朝右) - 两个三角形覆盖 y∈[-100,100], z∈[-100,100]
    scene.objects.push_back({ .v0={1, 0, 0}, .v1={1, 0, 600}, .v2={1, 600, 0}, .pos={0,0,0}, .color=red, .emission={0,0,0}, .rad=0, .tex_id=-1, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });

    // 右墙 (x=99, 蓝色, 法线朝左)
    scene.objects.push_back({ .v0={99, 0, 0}, .v1={99, 600, 0}, .v2={99, 0, 600}, .pos={0,0,0}, .color=blue, .emission={0,0,0}, .rad=0, .tex_id=-1, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });

    // 后墙 (z=0, 白色, 法线朝前, 带纹理)
    scene.objects.push_back({ .v0={-50, 0, 0}, .v1={150, 0, 0}, .v2={50, 300, 0}, .pos={0,0,0}, .color={1,1,1}, .emission={0,0,0}, .rad=0, .tex_id=0, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });
    // 前墙 (z=300, 白色, 法线朝前, 带纹理)
    scene.objects.push_back({ .v0={-50, 0, 300}, .v1={150, 0, 300}, .v2={50, 300, 300}, .pos={0,0,0}, .color={1,1,1}, .emission={0,0,0}, .rad=0, .tex_id=-1, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });

    // 地板 (y=0, 白色, 法线朝上)
    scene.objects.push_back({ .v0={-50, 0, 0}, .v1={150, 0, 0}, .v2={50, 0, 600}, .pos={0,0,0}, .color=white, .emission={0,0,0}, .rad=0, .tex_id=-1, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });

    // 天花板 (y=81.6, 白色, 法线朝下)
    scene.objects.push_back({ .v0={-50, 81.6, 0}, .v1={150, 81.6, 0}, .v2={50, 81.6, 600}, .pos={0,0,0}, .color=white, .emission={0,0,0}, .rad=0, .tex_id=-1, .fuzz=0, .padding=0, .refl=DIFF, .type=TRIANGLE, .pad2=0, .pad3=0 });

    // --- 外部模型 (Mesh) ---
    // 调用 loader 模块来加载 cube.obj 文件
    // 参数: 文件名, 目标容器, 位置偏移, 缩放大小, 颜色, 材质
    load_obj("assets/cube.obj", scene.objects,
             {50.0f, 10.0f, 130.0f}, // 位置: 放在盒子中间偏上
             10.0f,                 // 缩放: 模型原始大小是 -1到1，放大10倍
             gold,                  // 颜色: 金色
             SPEC,                  // 材质: 镜面
             0.9f);

    // --- 球体 (Sphere) ---
    scene.objects.push_back({
        .v0={0,0,0}, .v1={0,0,0}, .v2={0,0,0},
        .pos={73.0f, 16.5f, 78.0f},
        .color={0.99f, 0.99f, 0.99f},
        .emission={0,0,0},
        .rad=16.5f,
        .tex_id=-1,
        .fuzz = 0,
        .padding = 0,
        .refl=REFR,
        .type=SPHERE,
        .pad2=0, .pad3=0
    }); // 玻璃球 (右侧)

    scene.objects.push_back({
        .v0={0,0,0}, .v1={0,0,0}, .v2={0,0,0},
        .pos={27.0f, 16.5f, 47.0f},
        .color={0.99f, 0.99f, 0.99f},
        .emission={0,0,0},
        .rad=16.5f,
        .tex_id=-1,
        .fuzz = 1,
        .padding = 0,
        .refl=SPEC,
        .type=SPHERE,
        .pad2=0, .pad3=0
    }); // 镜面球 (右侧)

    // --- 光源 (Light) ---
    // 这是一个自发光的球体，模拟吸顶灯
    scene.objects.push_back({
        .v0={0,0,0}, .v1={0,0,0}, .v2={0,0,0},
        .pos={50.0f, 681.33f, 81.6f},
        .color={0,0,0},         // 灯的表面颜色是黑的 (不反射)
        .emission=light,        // 但它自己会发光
        .rad=600.0f,
        .tex_id=-1,
        .fuzz = 0,
        .padding = 0,
        .refl=DIFF,       // 灯本身是漫反射材质
        .type=SPHERE,
        .pad2=0, .pad3=0
    });

    // 打印场景信息
    printf("[Scene] Scene created with %lu objects.\n", scene.objects.size());
    printf("        Walls: 10 triangles, Mesh: ~triangles, Spheres: 3\n");

    return scene;
}
