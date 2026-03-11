#pragma once

#include <memory>
#include <string>
#include <vector>

/**
 * @file denoiser.h
 * @brief OpenVINO 帧降噪器接口
 *
 * 设计目标:
 * 1. 将 OpenVINO / NPU 相关细节封装到独立模块，避免主循环被推理代码淹没。
 * 2. 允许项目在“未检测到 NPU”或“NPU 初始化失败”时自动回退到原始渲染画面。
 * 3. 当前内置一个固定权重的轻量卷积降噪模型，方便学习 OpenVINO 推理链路。
 * 4. 通过清晰的接口边界，让后续替换成训练好的模型或加入异步推理更容易。
 */
class FrameDenoiser {
public:
    /// 构造函数不会立刻加载模型，真正的初始化发生在 initialize()。
    FrameDenoiser();
    ~FrameDenoiser();

    FrameDenoiser(FrameDenoiser&&) noexcept;
    FrameDenoiser& operator=(FrameDenoiser&&) noexcept;

    FrameDenoiser(const FrameDenoiser&) = delete;
    FrameDenoiser& operator=(const FrameDenoiser&) = delete;

    /// 初始化内置 NPU 降噪模型，并完成 OpenVINO 编译。
    void initialize(int frame_width, int frame_height);

    /// 是否已成功加载模型并可执行推理。
    bool is_enabled() const;

    /// 当前是否持有一帧可直接显示的降噪输出。
    bool has_output() const;

    /// 根据当前帧号判断这一帧是否应触发一次推理。
    /// 这个接口是给“同步版降噪流程”预留的；异步版当前直接每次都可提交。
    bool should_run(int frame_index) const;

    /// 当前帧实际采用多大比例的降噪结果做显示混合。
    /// 当前主程序没有使用这个混合值，但它保留了“以后做平滑过渡显示”的能力。
    float blend_alpha(int frame_index) const;

    /**
     * @brief 对一帧线性 RGB 图像执行降噪
     * @param input_rgb_nhwc 输入图像，布局固定为 NHWC，数值建议为线性 RGB
     * @param frame_width 当前渲染宽度
     * @param frame_height 当前渲染高度
     * @return true 表示本次推理成功并刷新了输出缓存
     */
    bool run(const float* input_rgb_nhwc, int frame_width, int frame_height);

    /// 当相机移动导致累积帧失效时，丢弃上一帧降噪结果。
    void reset();

    /// 获取最近一次成功推理的输出图像。
    const std::vector<float>& output_rgb() const;

    /// 获取当前模块状态，适合直接打印到日志/终端。
    const std::string& status() const;

    /// 获取当前实际请求的 OpenVINO 设备名。
    const std::string& device_name() const;

private:
    // PImpl 写法：把 OpenVINO 头文件和复杂实现细节藏到 cpp 里，
    // 减少头文件污染，也让编译依赖更清晰。
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
