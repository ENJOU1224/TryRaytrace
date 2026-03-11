#include "scene.h"
#include "loader.h" 

void add_quad(Scene& scene, Vec p0, Vec p1, Vec p2, Vec p3, Vec color, Vec emission, float metallic, float roughness) {
    scene.objects.push_back(make_object(p0, p1, p2, color, emission, metallic, roughness));
    scene.objects.push_back(make_object(p0, p2, p3, color, emission, metallic, roughness));
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
    scene.objects.push_back(make_object({-50,0,0}, {50,0,600}, {150,0,0}, white, black, wall_m, wall_r));
    // 2. 天花板
    scene.objects.push_back(make_object({-50,100,0}, {150,100,0}, {50,100,600}, white, black, wall_m, wall_r));
    // 3. 后墙 (远端)
    scene.objects.push_back(make_object({-50,0,0}, {150,0,0}, {50,200,0}, white, black, wall_m, wall_r, 1.45f, 0.0f, 0));
    // 4. 前墙 (近端，防止漏光)
    scene.objects.push_back(make_object({-50,0,300}, {150,0,300}, {50,200,300}, white, black, wall_m, wall_r));
    // 5. 左墙
    scene.objects.push_back(make_object({0,0,-50}, {0,200,50}, {0,0,550}, red, black, wall_m, wall_r));
    // 6. 右墙
    scene.objects.push_back(make_object({100,0,550}, {100,200,50}, {100,0,-50}, green, black, wall_m, wall_r));
    // 7. 灯
    scene.objects.push_back(make_object({30,99.9,30}, {70,99.9,30}, {50,99.9,50}, black, light_color, wall_m, wall_r));

    // 茶壶模型：
    // 现在把茶壶加回来，但保留“粗糙非金属墙面纯漫反射”那条修正。
    // 这样可以验证当前剩下的高亮异常，是否已经收敛成茶壶附近的局部镜面高方差。
    load_obj("assets/teapot.obj", scene.objects,
             {50.0f, 10.0f, 50.0f},
             10.0f, white,
             1.0f,
             0.0f);

    printf("[Scene] Scene created with %lu objects.\n", scene.objects.size());
    scene.world_bound = AABB::empty();
    for (const auto& obj : scene.objects) {
      scene.world_bound.grow(obj.v0);
      scene.world_bound.grow(obj.v0 + obj.edge1);
      scene.world_bound.grow(obj.v0 + obj.edge2);
    }
    scene.world_bound.min = scene.world_bound.min - make_vec(0.1f, 0.1f, 0.1f);
    scene.world_bound.max = scene.world_bound.max + make_vec(0.1f, 0.1f, 0.1f);
    return scene;
}
