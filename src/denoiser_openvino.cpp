#include "denoiser.h"

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * 降噪模型最常见的两种张量布局。
 * 项目内部统一向降噪器提供 NHWC，OpenVINO 负责按模型布局做转换。
 */
enum class TensorLayout {
    NCHW,
    NHWC,
};

/// 读取环境变量，不存在时返回空字符串，避免主流程里充斥 nullptr 判断。
std::string getenv_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

/// 解析正整数环境变量。配置非法时回退到默认值，而不是让程序直接崩掉。
int parse_positive_int(const std::string& text, int fallback) {
    if (text.empty()) {
        return fallback;
    }
    try {
        int value = std::stoi(text);
        return value > 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

/// 环境变量里的布局名允许大小写混用，这里统一转成大写。
std::string normalize_layout_name(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return text;
}

TensorLayout layout_from_name(const std::string& layout_name) {
    const std::string normalized = normalize_layout_name(layout_name);
    if (normalized == "NCHW") {
        return TensorLayout::NCHW;
    }
    if (normalized == "NHWC") {
        return TensorLayout::NHWC;
    }
    throw std::runtime_error("只支持 NCHW 或 NHWC 布局。");
}

/// 根据模型 shape 猜输入/输出布局，优先识别通道维是否为 3。
TensorLayout infer_layout_from_shape(const ov::PartialShape& shape, TensorLayout fallback) {
    if (shape.rank().is_static() && shape.rank().get_length() == 4) {
        if (shape[1].is_static() && shape[1].get_length() == 3) {
            return TensorLayout::NCHW;
        }
        if (shape[3].is_static() && shape[3].get_length() == 3) {
            return TensorLayout::NHWC;
        }
    }
    return fallback;
}

ov::Layout to_ov_layout(TensorLayout layout) {
    return ov::Layout(layout == TensorLayout::NCHW ? "NCHW" : "NHWC");
}

/**
 * 推理结果只做“安全化”，不提前压到 [0, 1]。
 * 这样既能过滤 NaN / Inf，也不会把潜在的高亮 HDR 信息过早裁掉。
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

/// 用于把模型输出重新映射回窗口分辨率，避免要求模型分辨率必须与渲染分辨率完全一致。
void resize_rgb_bilinear(const float* src,
                         int src_width,
                         int src_height,
                         float* dst,
                         int dst_width,
                         int dst_height) {
    if (src_width == dst_width && src_height == dst_height) {
        std::memcpy(dst, src, static_cast<size_t>(src_width) * src_height * 3 * sizeof(float));
        return;
    }

    const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);

    for (int y = 0; y < dst_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, src_height - 1);
        const int y1 = std::min(y0 + 1, src_height - 1);
        const float wy = src_y - static_cast<float>(y0);

        for (int x = 0; x < dst_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, src_width - 1);
            const int x1 = std::min(x0 + 1, src_width - 1);
            const float wx = src_x - static_cast<float>(x0);

            const size_t dst_base = static_cast<size_t>(y * dst_width + x) * 3;
            const size_t idx00 = static_cast<size_t>(y0 * src_width + x0) * 3;
            const size_t idx01 = static_cast<size_t>(y0 * src_width + x1) * 3;
            const size_t idx10 = static_cast<size_t>(y1 * src_width + x0) * 3;
            const size_t idx11 = static_cast<size_t>(y1 * src_width + x1) * 3;

            for (int c = 0; c < 3; ++c) {
                const float top = src[idx00 + c] * (1.0f - wx) + src[idx01 + c] * wx;
                const float bottom = src[idx10 + c] * (1.0f - wx) + src[idx11 + c] * wx;
                dst[dst_base + c] = top * (1.0f - wy) + bottom * wy;
            }
        }
    }
}

}  // namespace

/**
 * 具体实现体放在 .cpp 里，目的是:
 * 1. 隐藏 OpenVINO 头文件和实现细节，减少编译耦合。
 * 2. 给后续异步化、缓存多路 infer request 留空间。
 */
struct FrameDenoiser::Impl {
    bool enabled = false;
    bool has_output = false;
    int frame_width = 0;
    int frame_height = 0;
    int infer_interval = 1;
    int model_output_width = 0;
    int model_output_height = 0;
    std::string model_path;
    std::string device = "NPU";
    std::string status;
    std::vector<float> output_rgb;

    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    /// 根据环境变量加载模型并配置预处理/后处理。
    void initialize_from_env(int width, int height) {
        frame_width = width;
        frame_height = height;
        enabled = false;
        has_output = false;
        output_rgb.clear();

        model_path = getenv_or_empty("TRYRAYTRACE_DENOISE_MODEL");
        device = getenv_or_empty("TRYRAYTRACE_DENOISE_DEVICE");
        if (device.empty()) {
            device = "NPU";
        }
        infer_interval = parse_positive_int(getenv_or_empty("TRYRAYTRACE_DENOISE_INTERVAL"), 1);

        if (model_path.empty()) {
            status = "[降噪] 未设置 TRYRAYTRACE_DENOISE_MODEL，当前使用原始渲染画面。";
            return;
        }

        if (!std::filesystem::exists(model_path)) {
            status = "[降噪] 模型文件不存在：" + model_path + "，当前使用原始渲染画面。";
            return;
        }

        try {
            // 第一步：读取模型并做最基本的结构约束检查。
            // 这里故意只支持“单输入 + 单输出”的图像模型，
            // 目的是先把渲染管线接通，避免一开始就引入多输入辅助特征图的复杂度。
            auto model = core.read_model(model_path);
            if (model->inputs().size() != 1 || model->outputs().size() != 1) {
                throw std::runtime_error("当前只支持单输入单输出的图像降噪模型。");
            }

            // 第二步：确定模型输入/输出布局。
            // 项目内部统一使用 NHWC 连续内存，模型若是 NCHW，则交给 OpenVINO 自动插入转置。
            TensorLayout input_layout = TensorLayout::NCHW;
            const std::string input_layout_env = getenv_or_empty("TRYRAYTRACE_DENOISE_INPUT_LAYOUT");
            if (!input_layout_env.empty()) {
                input_layout = layout_from_name(input_layout_env);
            } else {
                input_layout = infer_layout_from_shape(model->input().get_partial_shape(), TensorLayout::NCHW);
            }

            TensorLayout output_layout = input_layout;
            const std::string output_layout_env = getenv_or_empty("TRYRAYTRACE_DENOISE_OUTPUT_LAYOUT");
            if (!output_layout_env.empty()) {
                output_layout = layout_from_name(output_layout_env);
            } else {
                output_layout = infer_layout_from_shape(model->output().get_partial_shape(), input_layout);
            }

            // 第三步：构建 OpenVINO 的预处理/后处理链。
            // 这里做的事情分别是：
            // 1. 告诉 Runtime：应用侧喂进来的是 NHWC、float32、分辨率为当前窗口大小。
            // 2. 告诉 Runtime：模型内部真正期望的布局是 NCHW 还是 NHWC。
            // 3. 若模型输入分辨率与窗口不一致，自动插入 resize。
            // 4. 模型输出无论内部布局如何，最终都转回 NHWC float32，方便 SDL 显示。
            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input().tensor()
                .set_element_type(ov::element::f32)
                .set_layout("NHWC")
                .set_shape(ov::PartialShape{1, height, width, 3});
            ppp.input().model().set_layout(to_ov_layout(input_layout));
            ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

            ppp.output().model().set_layout(to_ov_layout(output_layout));
            ppp.output().postprocess().convert_layout(ov::Layout("NHWC")).convert_element_type(ov::element::f32);
            ppp.output().tensor().set_layout("NHWC").set_element_type(ov::element::f32);

            model = ppp.build();

            // 第四步：将模型真正编译到目标设备。
            // 默认目标是 NPU；如果 NPU 不支持当前模型，这一步会抛异常并自动回退。
            compiled_model = core.compile_model(model, device);
            infer_request = compiled_model.create_infer_request();

            const ov::Shape output_shape = infer_request.get_output_tensor().get_shape();
            if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[3] != 3) {
                throw std::runtime_error("模型输出不是 1xHxWx3 的 RGB 图像。");
            }

            model_output_height = static_cast<int>(output_shape[1]);
            model_output_width = static_cast<int>(output_shape[2]);
            output_rgb.resize(static_cast<size_t>(frame_width) * frame_height * 3);
            enabled = true;

            std::ostringstream oss;
            oss << "[降噪] 已加载模型：" << model_path
                << " | 设备：" << device
                << " | 推理间隔：" << infer_interval << " 帧";
            if (model_output_width != frame_width || model_output_height != frame_height) {
                oss << " | 模型输出：" << model_output_width << "x" << model_output_height
                    << "，显示时会缩放回 " << frame_width << "x" << frame_height;
            }
            status = oss.str();
        } catch (const std::exception& ex) {
            status = std::string("[降噪] 初始化失败，已回退到原始画面：") + ex.what();
            enabled = false;
            has_output = false;
        }
    }

    bool should_run(int frame_index) const {
        if (!enabled) {
            return false;
        }
        return !has_output || (frame_index % infer_interval == 0);
    }

    bool run(const float* input_rgb_nhwc, int width, int height) {
        if (!enabled) {
            return false;
        }
        if (width != frame_width || height != frame_height) {
            status = "[降噪] 当前实现只支持固定渲染分辨率，分辨率变化后已关闭降噪。";
            enabled = false;
            has_output = false;
            return false;
        }

        try {
            // 每一帧都直接把应用侧的线性 RGB 缓冲区包装成 OpenVINO Tensor。
            // 这里不额外复制输入数据，目的是让主循环里的 CPU 开销尽量小。
            ov::Tensor input_tensor(ov::element::f32,
                                    ov::Shape{1, static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                    const_cast<float*>(input_rgb_nhwc));
            infer_request.set_input_tensor(input_tensor);
            infer_request.infer();

            const ov::Tensor result = infer_request.get_output_tensor();
            const float* result_data = result.data<const float>();
            if (model_output_width == frame_width && model_output_height == frame_height) {
                // 输出分辨率与窗口一致时，直接拷贝即可。
                std::memcpy(output_rgb.data(),
                            result_data,
                            output_rgb.size() * sizeof(float));
            } else {
                // 如果模型输出的是固定小分辨率，这里统一放大回窗口尺寸。
                resize_rgb_bilinear(result_data,
                                    model_output_width,
                                    model_output_height,
                                    output_rgb.data(),
                                    frame_width,
                                    frame_height);
            }

            sanitize_rgb(output_rgb);
            has_output = true;
            return true;
        } catch (const std::exception& ex) {
            status = std::string("[降噪] 推理失败，已回退到原始画面：") + ex.what();
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

void FrameDenoiser::initialize_from_env(int frame_width, int frame_height) {
    impl_->initialize_from_env(frame_width, frame_height);
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
    return impl_->device;
}
