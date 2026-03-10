#include "scene.h"
#include "loader.h" 

void add_quad(Scene& scene, Vec p0, Vec p1, Vec p2, Vec p3, Vec color, Vec emission, float metallic, float roughness) {
    scene.objects.push_back({
        .v0=p0, .v1=p1, .v2=p2, .albedo=color, .emission=emission, .metallic=metallic, .roughness=roughness, .ior=1.45f, .transmission=0.0f, .tex_id=-1
    });
    scene.objects.push_back({
        .v0=p0, .v1=p2, .v2=p3, .albedo=color, .emission=emission, .metallic=metallic, .roughness=roughness, .ior=1.45f, .transmission=0.0f, .tex_id=-1
    });
}

Scene create_cornell_box() {
    Scene scene;
    scene.texture_files.push_back("assets/earth.ppm");

    Vec white = {0.75f, 0.75f, 0.75f};
    Vec red   = {0.75f, 0.25f, 0.25f};
    Vec green = {0.25f, 0.75f, 0.25f};
    Vec light_color = {20, 20, 20};
    Vec black = {0,0,0};

    // 墙壁：标准漫反射
    float wall_m = 0.0f;
    float wall_r = 1.0f;

    // 1. 地板
    scene.objects.push_back({.v0={-50,0,0}, .v1={50,0,600}, .v2={150,0,0}, .albedo=white, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});
    // 2. 天花板
    scene.objects.push_back({.v0={-50,100,0}, .v1={150,100,0}, .v2={50,100,600}, .albedo=white, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});
    // 3. 后墙 (远端)
    scene.objects.push_back({.v0={-50,0,0}, .v1={150,0,0}, .v2={50,200,0}, .albedo=white, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=0});
    // 4. 前墙 (近端，防止漏光)
    scene.objects.push_back({.v0={-50,0,300}, .v1={150,0,300}, .v2={50,200,300}, .albedo=white, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});
    // 5. 左墙
    scene.objects.push_back({.v0={0,0,-50}, .v1={0,200,50}, .v2={0,0,550}, .albedo=red, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});
    // 6. 右墙
    scene.objects.push_back({.v0={100,0,550}, .v1={100,200,50}, .v2={100,0,-50}, .albedo=green, .emission=black, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});
    // 7. 灯
    scene.objects.push_back({.v0={30,99.9,30}, .v1={70,99.9,30}, .v2={50,99.9,50}, .albedo=black, .emission=light_color, .metallic=wall_m, .roughness=wall_r, .ior=1.45f, .transmission=0.0f, .tex_id=-1});

    // 茶壶模型：默认非镜面，如果想要镜面可以手动调参数
    load_obj("assets/teapot.obj", scene.objects, 
             {50.0f, 10.0f, 50.0f}, 
             10.0f, white, 
             1.0f, // metallic
             0.0f  // roughness (1.0 = 完全漫反射)
    );                 

    printf("[Scene] Scene created with %lu objects.\n", scene.objects.size());
    scene.world_bound = AABB::empty();
    for (const auto& obj : scene.objects) {
      scene.world_bound.grow(obj.v0); scene.world_bound.grow(obj.v1); scene.world_bound.grow(obj.v2);
    }
    scene.world_bound.min = scene.world_bound.min - make_vec(0.1f, 0.1f, 0.1f);
    scene.world_bound.max = scene.world_bound.max + make_vec(0.1f, 0.1f, 0.1f);
    return scene;
}
