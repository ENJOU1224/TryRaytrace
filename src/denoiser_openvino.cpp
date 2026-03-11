#include "denoiser.h"

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/opsets/opset13.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kDenoiseDeviceName = "NPU";
constexpr int kDenoiseInterval = 1;
constexpr int kMinimumFramesBeforeDenoise = 3;
constexpr int kBlendRampEndFrame = 12;
constexpr float kBlendAlphaMin = 0.18f;
constexpr float kBlendAlphaMax = 0.58f;
constexpr float kPreserveWeight = 0.82f;
constexpr float kBlurWeight = 0.18f;
constexpr float kFireflyGuardScale = 1.25f;
constexpr float kFireflyGuardBias = 0.02f;

/**
 * 将 OpenVINO 设备列表中是否存在 NPU 的判断逻辑集中起来。
 * 这样主流程里只需要关心“能不能启用降噪”，不用重复处理字符串匹配。
 */
bool has_npu_device(ov::Core& core) {
    const std::vector<std::string> devices = core.get_available_devices();
    for (const std::string& device : devices) {
        if (device.find("NPU") != std::string::npos) {
            return true;
        }
    }
    return false;
}

/**
 * 生成一个 5x5 高斯核。
 *
 * 这不是训练出来的参数，而是手工指定的平滑滤波器。
 * 选它的原因很务实：
 * 1. 卷积运算是 NPU 最擅长、也最稳定支持的一类算子。
 * 2. 代码里直接能看懂每个权重的含义，适合教学。
 * 3. 它对路径追踪中的高频噪点有立竿见影的抑制效果。
 */
std::vector<float> build_gaussian_kernel_5x5() {
    static constexpr float kRawKernel[25] = {
        1.0f,  4.0f,  6.0f,  4.0f, 1.0f,
        4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
        6.0f, 24.0f, 36.0f, 24.0f, 6.0f,
        4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
        1.0f,  4.0f,  6.0f,  4.0f, 1.0f,
    };

    std::vector<float> kernel(25);
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kRawKernel[i] / 256.0f;
    }
    return kernel;
}

/**
 * 生成卷积权重，形状为 [3, 3, 5, 5]。
 *
 * 每个输出通道只读取对应输入通道：
 * - R 只卷积 R
 * - G 只卷积 G
 * - B 只卷积 B
 *
 * 这样可以避免跨颜色通道串扰，先把问题收敛成“对每个颜色通道分别降噪”。
 */
std::vector<float> build_rgb_blur_weights() {
    const std::vector<float> gaussian = build_gaussian_kernel_5x5();
    std::vector<float> weights(3 * 3 * 5 * 5, 0.0f);

    for (int c = 0; c < 3; ++c) {
        const size_t base = static_cast<size_t>((c * 3 + c) * 25);
        std::copy(gaussian.begin(), gaussian.end(), weights.begin() + static_cast<long>(base));
    }
    return weights;
}

/// 构造 [1, 3, 1, 1] 常量，用于对三个颜色通道做同权重缩放。
std::vector<float> build_channel_scale(float value) {
    return std::vector<float>{value, value, value};
}

/**
 * 构建内置轻量降噪模型。
 *
 * 模型结构很简单：
 * input
 *   -> 5x5 Gaussian Conv -> blurred
 *   -> firefly guard: min(input, blurred * 1.25 + 0.02) -> guarded
 *   -> 5x5 Gaussian Conv(guarded) -> refined
 * guarded * 0.82 + refined * 0.18 -> output
 *
 * 这本质上是一个固定权重的“异常高亮保护 + 残差卷积”网络。
 * 它不是追求极限效果的 SOTA 模型，而是一个非常适合当前项目阶段的工程基线：
 * - 不依赖外部模型文件
 * - 算子集合极小，NPU 兼容性高
 * - 代码可读性强，适合学习 OpenVINO 图构建
 * - 对 fireflies 比单纯模糊更克制
 */
std::shared_ptr<ov::Model> build_builtin_denoise_model(int frame_height, int frame_width) {
    using namespace ov::opset13;

    auto input = std::make_shared<Parameter>(ov::element::f32,
                                             ov::Shape{1, 3, static_cast<size_t>(frame_height), static_cast<size_t>(frame_width)});

    const std::vector<float> blur_weights_data = build_rgb_blur_weights();
    auto blur_weights = std::make_shared<Constant>(ov::element::f32,
                                                   ov::Shape{3, 3, 5, 5},
                                                   blur_weights_data.data());

    auto blurred = std::make_shared<Convolution>(input->output(0),
                                                 blur_weights->output(0),
                                                 ov::Strides{1, 1},
                                                 ov::CoordinateDiff{2, 2},
                                                 ov::CoordinateDiff{2, 2},
                                                 ov::Strides{1, 1});

    const std::vector<float> firefly_scale_data = build_channel_scale(kFireflyGuardScale);
    auto firefly_scale = std::make_shared<Constant>(ov::element::f32,
                                                    ov::Shape{1, 3, 1, 1},
                                                    firefly_scale_data.data());

    const std::vector<float> firefly_bias_data = build_channel_scale(kFireflyGuardBias);
    auto firefly_bias = std::make_shared<Constant>(ov::element::f32,
                                                   ov::Shape{1, 3, 1, 1},
                                                   firefly_bias_data.data());

    auto firefly_limit_scaled = std::make_shared<Multiply>(blurred->output(0), firefly_scale->output(0));
    auto firefly_limit = std::make_shared<Add>(firefly_limit_scaled->output(0), firefly_bias->output(0));
    auto guarded = std::make_shared<Minimum>(input->output(0), firefly_limit->output(0));

    auto refined = std::make_shared<Convolution>(guarded->output(0),
                                                 blur_weights->output(0),
                                                 ov::Strides{1, 1},
                                                 ov::CoordinateDiff{2, 2},
                                                 ov::CoordinateDiff{2, 2},
                                                 ov::Strides{1, 1});

    const std::vector<float> preserve_scale_data = build_channel_scale(kPreserveWeight);
    auto preserve_scale = std::make_shared<Constant>(ov::element::f32,
                                                     ov::Shape{1, 3, 1, 1},
                                                     preserve_scale_data.data());

    const std::vector<float> blur_scale_data = build_channel_scale(kBlurWeight);
    auto blur_scale = std::make_shared<Constant>(ov::element::f32,
                                                 ov::Shape{1, 3, 1, 1},
                                                 blur_scale_data.data());

    auto preserved = std::make_shared<Multiply>(guarded->output(0), preserve_scale->output(0));
    auto refined_scaled = std::make_shared<Multiply>(refined->output(0), blur_scale->output(0));
    auto output = std::make_shared<Add>(preserved->output(0), refined_scaled->output(0));
    output->get_output_tensor(0).set_names({"denoised_rgb"});

    auto result = std::make_shared<Result>(output->output(0));
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "builtin_npu_denoiser");
}

/**
 * 推理结果做最基础的数值清理。
 * 这里不做 Gamma，也不主动裁掉高亮，只过滤明显错误值。
 */
void sanitize_rgb(std::vector<float>& buffer) {
    for (float& value : buffer) {
        if (std::isnan(value) || std::isinf(value)) {
            value = 0.0f;
        } else if (value < 0.0f) {
            value = 0.0f;
        }
    }
}

}  // namespace

struct FrameDenoiser::Impl {
    bool enabled = false;
    bool has_output = false;
    int frame_width = 0;
    int frame_height = 0;
    std::string status;
    std::vector<float> output_rgb;

    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    void initialize(int width, int height) {
        frame_width = width;
        frame_height = height;
        enabled = false;
        has_output = false;
        output_rgb.clear();

        if (!has_npu_device(core)) {
            status = "[降噪] 未检测到 OpenVINO NPU 设备，已禁用内置降噪。";
            return;
        }

        try {
            auto model = build_builtin_denoise_model(height, width);

            // 应用侧维持 NHWC 连续内存，模型内部使用更常见的 NCHW。
            // 这一步把布局转换交给 OpenVINO 的预处理图完成。
            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input().tensor()
                .set_element_type(ov::element::f32)
                .set_layout("NHWC")
                .set_shape(ov::PartialShape{1, height, width, 3});
            ppp.input().model().set_layout(ov::Layout("NCHW"));

            ppp.output().model().set_layout(ov::Layout("NCHW"));
            ppp.output().postprocess()
                .convert_layout(ov::Layout("NHWC"))
                .convert_element_type(ov::element::f32);
            ppp.output().tensor()
                .set_layout("NHWC")
                .set_element_type(ov::element::f32);

            model = ppp.build();
            compiled_model = core.compile_model(model, kDenoiseDeviceName);
            infer_request = compiled_model.create_infer_request();

            output_rgb.resize(static_cast<size_t>(frame_width) * frame_height * 3, 0.0f);
            enabled = true;
            status =
                "[降噪] 已启用内置 OpenVINO NPU 降噪器：5x5 高斯残差卷积，输入输出为当前窗口分辨率。";
        } catch (const std::exception& ex) {
            status = std::string("[降噪] NPU 降噪初始化失败，已回退到原始画面：") + ex.what();
            enabled = false;
            has_output = false;
        }
    }

    bool should_run(int frame_index) const {
        if (!enabled) {
            return false;
        }
        if (frame_index < kMinimumFramesBeforeDenoise) {
            return false;
        }
        return !has_output || (frame_index % kDenoiseInterval == 0);
    }

    bool run(const float* input_rgb_nhwc, int width, int height) {
        if (!enabled) {
            return false;
        }
        if (width != frame_width || height != frame_height) {
            status = "[降噪] 当前实现只支持固定渲染分辨率，检测到尺寸变化后已关闭降噪。";
            enabled = false;
            has_output = false;
            return false;
        }

        try {
            ov::Tensor input_tensor(ov::element::f32,
                                    ov::Shape{1, static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                    const_cast<float*>(input_rgb_nhwc));
            infer_request.set_input_tensor(input_tensor);
            infer_request.infer();

            const ov::Tensor result = infer_request.get_output_tensor();
            std::memcpy(output_rgb.data(), result.data<const float>(), output_rgb.size() * sizeof(float));
            sanitize_rgb(output_rgb);
            has_output = true;
            return true;
        } catch (const std::exception& ex) {
            status = std::string("[降噪] NPU 推理失败，已回退到原始画面：") + ex.what();
            enabled = false;
            has_output = false;
            return false;
        }
    }

    void reset() {
        has_output = false;
    }
};

FrameDenoiser::FrameDenoiser() : impl_(std::make_unique<Impl>()) {}

FrameDenoiser::~FrameDenoiser() = default;

FrameDenoiser::FrameDenoiser(FrameDenoiser&& other) noexcept = default;

FrameDenoiser& FrameDenoiser::operator=(FrameDenoiser&& other) noexcept = default;

void FrameDenoiser::initialize(int frame_width, int frame_height) {
    impl_->initialize(frame_width, frame_height);
}

bool FrameDenoiser::is_enabled() const {
    return impl_ && impl_->enabled;
}

bool FrameDenoiser::has_output() const {
    return impl_ && impl_->has_output;
}

bool FrameDenoiser::should_run(int frame_index) const {
    return impl_ && impl_->should_run(frame_index);
}

float FrameDenoiser::blend_alpha(int frame_index) const {
    if (!impl_ || !impl_->enabled || frame_index < kMinimumFramesBeforeDenoise) {
        return 0.0f;
    }

    if (frame_index >= kBlendRampEndFrame) {
        return kBlendAlphaMax;
    }

    const float t = static_cast<float>(frame_index - kMinimumFramesBeforeDenoise) /
                    static_cast<float>(std::max(1, kBlendRampEndFrame - kMinimumFramesBeforeDenoise));
    return kBlendAlphaMin + (kBlendAlphaMax - kBlendAlphaMin) * std::clamp(t, 0.0f, 1.0f);
}

bool FrameDenoiser::run(const float* input_rgb_nhwc, int frame_width, int frame_height) {
    return impl_ && impl_->run(input_rgb_nhwc, frame_width, frame_height);
}

void FrameDenoiser::reset() {
    if (impl_) {
        impl_->reset();
    }
}

const std::vector<float>& FrameDenoiser::output_rgb() const {
    return impl_->output_rgb;
}

const std::string& FrameDenoiser::status() const {
    return impl_->status;
}

const std::string& FrameDenoiser::device_name() const {
    static const std::string kDeviceName = kDenoiseDeviceName;
    return kDeviceName;
}
