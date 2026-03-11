#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

namespace {

// 用 specialization constant 约束当前 probe 只需要三角形特性，
// 这样更贴近 Embree 官方 minimal_sycl 示例，也能减少 GPU JIT 负担。
const sycl::specialization_id<RTCFeatureFlags> g_feature_mask;
constexpr RTCFeatureFlags kRequiredFeatures = RTC_FEATURE_FLAG_TRIANGLE;

struct ProbeResult {
    uint32_t geom_id = RTC_INVALID_GEOMETRY_ID;
    uint32_t prim_id = RTC_INVALID_GEOMETRY_ID;
    float tfar = std::numeric_limits<float>::infinity();
    int occluded = 0;
};

const char* backend_name(sycl::backend backend) {
    switch (backend) {
        case sycl::backend::ext_oneapi_level_zero:
            return "level_zero";
        case sycl::backend::opencl:
            return "opencl";
        case sycl::backend::ext_oneapi_cuda:
            return "cuda";
        case sycl::backend::ext_oneapi_hip:
            return "hip";
        case sycl::backend::all:
            return "all";
        default:
            return "unknown";
    }
}

void error_function(void*, RTCError error, const char* message) {
    std::cerr << "[Embree] " << rtcGetErrorString(error) << ": "
              << (message ? message : "(no message)") << std::endl;
}

RTCDevice create_embree_device(const sycl::context& context, const sycl::device& device) {
    RTCDevice embree_device = rtcNewSYCLDevice(context, "");
    if (!embree_device) {
        const RTCError error = rtcGetDeviceError(nullptr);
        const char* message = rtcGetDeviceLastErrorMessage(nullptr);
        throw std::runtime_error(
            std::string("rtcNewSYCLDevice failed: ") +
            rtcGetErrorString(error) +
            (message ? std::string(" | ") + message : std::string()));
    }

    rtcSetDeviceSYCLDevice(embree_device, device);
    rtcSetDeviceErrorFunction(embree_device, error_function, nullptr);
    return embree_device;
}

void print_detected_devices() {
    const auto platforms = sycl::platform::get_platforms();
    std::cout << "[EmbreeProbe] Detected SYCL devices:" << std::endl;
    for (const auto& platform : platforms) {
        for (const auto& device : platform.get_devices()) {
            bool supported = false;
            try {
                supported = rtcIsSYCLDeviceSupported(device);
            } catch (...) {
                supported = false;
            }

            std::cout << "  - " << device.get_info<sycl::info::device::name>()
                      << " | backend=" << backend_name(platform.get_backend())
                      << " | gpu=" << (device.is_gpu() ? "yes" : "no")
                      << " | embree_supported=" << (supported ? "yes" : "no")
                      << std::endl;
        }
    }
}

RTCScene create_triangle_scene(RTCDevice device, const sycl::queue& queue) {
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = sycl::malloc_shared<float>(9, queue);
    uint32_t* indices = sycl::malloc_shared<uint32_t>(3, queue);
    if (!vertices || !indices) {
        throw std::bad_alloc();
    }

    // 一个位于 z=0 平面上的简单三角形。
    // 这个场景极小，目的是验证“最小 Embree GPU 查询链是否工作”，
    // 而不是验证复杂材质或多物体场景。
    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;
    vertices[6] = 0.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;

    rtcSetSharedGeometryBuffer(geometry,
                               RTC_BUFFER_TYPE_VERTEX,
                               0,
                               RTC_FORMAT_FLOAT3,
                               vertices,
                               0,
                               3 * sizeof(float),
                               3);
    rtcSetSharedGeometryBuffer(geometry,
                               RTC_BUFFER_TYPE_INDEX,
                               0,
                               RTC_FORMAT_UINT3,
                               indices,
                               0,
                               3 * sizeof(uint32_t),
                               1);

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene);
    return scene;
}

void run_probe_kernel(sycl::queue& queue, RTCTraversable traversable, ProbeResult* result) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.set_specialization_constant<g_feature_mask>(kRequiredFeatures);

        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1>, sycl::kernel_handler kh) {
            // 先做一次最近交点查询，再做一次遮挡查询。
            // 如果这两个 API 都能在 GPU 上正确工作，说明 Embree GPU 链路已经通了。
            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            args.feature_mask = kh.get_specialization_constant<g_feature_mask>();

            RTCRayHit rayhit;
            rayhit.ray.org_x = 0.25f;
            rayhit.ray.org_y = 0.25f;
            rayhit.ray.org_z = -1.0f;
            rayhit.ray.dir_x = 0.0f;
            rayhit.ray.dir_y = 0.0f;
            rayhit.ray.dir_z = 1.0f;
            rayhit.ray.tnear = 0.0f;
            rayhit.ray.tfar = std::numeric_limits<float>::infinity();
            rayhit.ray.mask = 0xFFFFFFFFu;
            rayhit.ray.flags = 0;
            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            rtcTraversableIntersect1(traversable, &rayhit, &args);

            result->geom_id = rayhit.hit.geomID;
            result->prim_id = rayhit.hit.primID;
            result->tfar = rayhit.ray.tfar;

            RTCOccludedArguments occluded_args;
            rtcInitOccludedArguments(&occluded_args);
            occluded_args.feature_mask = kh.get_specialization_constant<g_feature_mask>();

            RTCRay shadow_ray;
            shadow_ray.org_x = 0.25f;
            shadow_ray.org_y = 0.25f;
            shadow_ray.org_z = -1.0f;
            shadow_ray.dir_x = 0.0f;
            shadow_ray.dir_y = 0.0f;
            shadow_ray.dir_z = 1.0f;
            shadow_ray.tnear = 0.0f;
            shadow_ray.tfar = std::numeric_limits<float>::infinity();
            shadow_ray.mask = 0xFFFFFFFFu;
            shadow_ray.flags = 0;

            rtcTraversableOccluded1(traversable, &shadow_ray, &occluded_args);
            result->occluded = shadow_ray.tfar < 0.0f ? 1 : 0;
        });
    });
    queue.wait_and_throw();
}

}  // namespace

int main() {
    try {
        // 启动时先枚举一遍当前系统里的 SYCL 设备，并打印 Embree 是否支持。
        // 这样一旦失败，日志里也能直接看到“设备根本没被识别”还是“设备存在但创建失败”。
        print_detected_devices();
        sycl::device device(rtcSYCLDeviceSelector);
        sycl::context context(device);
        sycl::queue queue(device);

        RTCDevice embree_device = create_embree_device(context, device);
        RTCScene scene = create_triangle_scene(embree_device, queue);
        RTCTraversable traversable = rtcGetSceneTraversable(scene);

        ProbeResult* result = sycl::malloc_shared<ProbeResult>(1, queue);
        if (!result) {
            throw std::bad_alloc();
        }
        *result = {};

        run_probe_kernel(queue, traversable, result);

        std::cout << "[EmbreeProbe] SYCL device: "
                  << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "[EmbreeProbe] geom=" << result->geom_id
                  << " prim=" << result->prim_id
                  << " tfar=" << result->tfar
                  << " occluded=" << result->occluded << std::endl;

        const bool hit_ok =
            result->geom_id != RTC_INVALID_GEOMETRY_ID &&
            result->prim_id != RTC_INVALID_GEOMETRY_ID &&
            result->tfar > 0.0f;
        if (!hit_ok) {
            std::cerr << "[EmbreeProbe] Ray query did not hit the expected test triangle." << std::endl;
            sycl::free(result, queue);
            rtcReleaseScene(scene);
            rtcReleaseDevice(embree_device);
            return 2;
        }

        sycl::free(result, queue);
        rtcReleaseScene(scene);
        rtcReleaseDevice(embree_device);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[EmbreeProbe] Exception: " << ex.what() << std::endl;
        print_detected_devices();
        return 1;
    }
}
