#include "scene.h"
#include "loader.h" // 引入 OBJ 加载器

// 辅助函数: 添加一个矩形 (2个三角形)
// p0, p1, p2, p3 逆时针顺序
void add_quad(Scene& scene, Vec p0, Vec p1, Vec p2, Vec p3, Vec color, Vec emission, Refl_t refl) {
    scene.objects.push_back({
        .v0=p0, .v1=p1, .v2=p2, 
    });
    scene.objects.push_back({
        .v0=p0, .v1=p2, .v2=p3, 
    });
}
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
    Vec green = {0.25f, 0.75f, 0.25f}; // 原版Cornell Box右墙是绿的，你之前是蓝的，看你喜好
    Vec light_color = {20, 20, 20};
    Vec black = {0,0,0};

    // 辅助 lambda: 快速生成 PBR 材质参数
    // 默认是粗糙的非金属
    auto make_mat = [](float m, float r, float t = 0.0f, float ior = 1.45f) {
        // 返回 metallic, roughness, ior, transmission
        struct { float m, r, i, t; } p = {m, r, ior, t};
        return p;
    };

    // ------------------------------------------------------------------
    // 3. 物体构建 (Procedural Scene Generation)
    // ------------------------------------------------------------------
    // 墙壁 (粗糙非金属)
    auto wall_mat = make_mat(0.0f, 1.0f);
    
    // 1. 地板 (y=0)
    scene.objects.push_back({
        .v0={-50,0,0}, .v1={50,0,600}, .v2={150,0,0}, 
        .albedo=white, .emission=black, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=-1});
    
    // 2. 天花板 (y=100)
    scene.objects.push_back({
        .v0={-50,100,0}, .v1={150,100,0}, .v2={50,100,600}, 
        .albedo=white, .emission=black, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=-1});

    // 3. 后墙 (z=0)
    scene.objects.push_back({
        .v0={-50,0,0}, .v1={150,0,0}, .v2={50,200,0}, 
        .albedo=white, .emission=black, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=0});

    // 3. 后墙 (z=0)
    scene.objects.push_back({
        .v0={-50,0,300}, .v1={150,0,300}, .v2={50,200,300}, 
        .albedo=black, .emission=black, .metallic=1, .roughness=0, .ior=0, .transmission=0, .tex_id=-1});

    // 4. 左墙 (x=0, 红)
    scene.objects.push_back({
        .v0={0,0,-50}, .v1={0,200,50}, .v2={0,0,550}, 
        .albedo=red, .emission=black, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=-1});

    // 5. 右墙 (x=100, 绿)
    scene.objects.push_back({
        .v0={100,0,550}, .v1={100,200,50}, .v2={100,0,-50}, 
        .albedo=green, .emission=black, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=-1});

    // 6. 顶灯 (天花板中间的一个方块)
    scene.objects.push_back({
        .v0={30,99.9,30}, .v1={70,99.9,30}, .v2={50,99.9,50}, 
        .albedo=black, .emission=light_color, .metallic=wall_mat.m, .roughness=wall_mat.r, .ior=wall_mat.i, .transmission=wall_mat.t, .tex_id=-1});

    // --- 外部模型 (Mesh) ---
    // 调用 loader 模块来加载 cube.obj 文件
    // 参数: 文件名, 目标容器, 位置偏移, 缩放大小, 颜色, 材质
    load_obj("assets/cow.obj", scene.objects, 
             {50.0f, 25.0f, 50.0f}, // 位置: 放在盒子中间偏上
             8.0f,                 // 缩放: 模型原始大小是 -1到1，放大10倍
             white,    // 颜色: 金色
             0.0f,                  // 材质: 镜面
             0.0f);                 

    // 打印场景信息
    printf("[Scene] Scene created with %lu objects.\n", scene.objects.size());

    scene.world_bound = AABB::empty();
    
    for (const auto& obj : scene.objects) {
      scene.world_bound.grow(obj.v0);
      scene.world_bound.grow(obj.v1);
      scene.world_bound.grow(obj.v2);
    }

    // 稍微把盒子往外扩一点点 (Epsilon)，防止刚好贴边导致误判
    scene.world_bound.min = scene.world_bound.min - make_vec(0.1f, 0.1f, 0.1f);
    scene.world_bound.max = scene.world_bound.max + make_vec(0.1f, 0.1f, 0.1f);

    printf("[Scene] World Bound: Min(%.1f, %.1f, %.1f) Max(%.1f, %.1f, %.1f)\n",
           scene.world_bound.min.x, scene.world_bound.min.y, scene.world_bound.min.z,
           scene.world_bound.max.x, scene.world_bound.max.y, scene.world_bound.max.z);

    return scene;
}
