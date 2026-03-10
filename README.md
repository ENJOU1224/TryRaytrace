# Lunar-Raytrace: 高性能 SYCL 光线追踪引擎

这是一个针对 **Intel Lunar Lake (Core Ultra 258V)** 架构深度优化的路径追踪渲染器。它成功地将原本的 NVIDIA CUDA 后端迁移到了现代的 **Intel oneAPI (SYCL)** 架构，实现了在集成核显 (Arc 140V) 上的高性能硬件加速。

## 🚀 核心特性

- **跨架构支持**: 基于 SYCL 标准编写，可在 Intel GPU (Level Zero/OpenCL) 和 CPU 上无缝切换。
- **Lunar Lake 优化**:
  - **USM (Unified Shared Memory)**: 利用 SoC 统一内存特性，实现 CPU/GPU 零拷贝数据传输。
  - **Xe2 矢量优化**: 关键路径使用 `sycl::native` 指令集和 FMA (乘加) 指令。
  - **SIMD 调度**: 针对 Xe2 Sub-group 16/32 优化的线程组布局 (16x8)。
- **物理渲染算法**:
  - **BVH 加速结构**: 线性化 BVH 树，支持高速射线求交。
  - **NEE (Next Event Estimation)**: 显式光源采样，大幅降低阴影区域噪点。
  - **Pro 级 PBR 材质**: 支持金属 (Specular)、玻璃 (Refraction) 和漫反射 (Diffuse)，基于 Schlick 菲涅尔近似。
- **稳定性工程**:
  - **Firefly Clamping**: 严格的数值钳制，防止出现高亮异常噪点。
  - **鲁棒性防火墙**: 内置 NaN/Inf 过滤与负值清理。
  - **实时反馈**: 实时 FPS 统计、终端渲染进度刷新及自动 Snapshot/Log 保存。
- **NPU 帧降噪接口**:
  - 基于 OpenVINO C++ Runtime，可将单输入单输出 RGB 图像模型部署到 `NPU`。
  - 当前实现支持 `NCHW/NHWC` 两类 4D 图像张量，并在显示前对输出做布局转换与缩放。
  - 未配置模型时，渲染器会自动回退到原始画面，不影响主渲染流程。

## 🛠️ 环境需求

- **硬件**: Intel Core Ultra 系列 (推荐 258V)
- **系统**: Linux (如 Ubuntu 24.04)
- **驱动**: 
  - `intel-level-zero-gpu` (推荐)
  - `intel-opencl-icd`
- **工具链**: [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (需包含 `icpx` 编译器)
- **OpenVINO**: 需提供 C++ Runtime 与 NPU 插件

## 🔨 构建与运行

1. **激活 oneAPI 与 OpenVINO 环境**:
   ```bash
   source ~/intel/oneapi/setvars.sh
   source ~/intel/openvino_2026/setupvars.sh
   ```

2. **编译项目**:
   ```bash
   make -j$(nproc)
   ```

3. **运行渲染器**:
   ```bash
    ./bin/sycl_engine
   ```

4. **检查 OpenVINO 是否识别到 NPU**:
   ```bash
   make checknpu
   ```

## 🧠 NPU 降噪使用方式

渲染器不会自带降噪模型。要启用 NPU 帧降噪，请在运行前设置模型路径：

```bash
export TRYRAYTRACE_DENOISE_MODEL=/path/to/your_denoise_model.xml
export TRYRAYTRACE_DENOISE_DEVICE=NPU
export TRYRAYTRACE_DENOISE_INTERVAL=1
./bin/sycl_engine
```

可选环境变量：

- `TRYRAYTRACE_DENOISE_MODEL`: OpenVINO IR/ONNX 模型路径，未设置时自动关闭降噪。
- `TRYRAYTRACE_DENOISE_DEVICE`: 推理设备，默认 `NPU`。
- `TRYRAYTRACE_DENOISE_INTERVAL`: 每隔多少帧执行一次降噪，默认 `1`。
- `TRYRAYTRACE_DENOISE_INPUT_LAYOUT`: 手动指定模型输入布局，支持 `NCHW` / `NHWC`。
- `TRYRAYTRACE_DENOISE_OUTPUT_LAYOUT`: 手动指定模型输出布局，支持 `NCHW` / `NHWC`。

模型约束：

- 仅支持单输入单输出模型。
- 输入输出都应表示 3 通道 RGB 图像。
- 如果模型输出分辨率与窗口分辨率不同，程序会在显示前缩放回当前窗口大小。

## 🎮 操作快捷键

- **WASD**: 移动相机
- **鼠标**: 旋转视角
- **P 键**: 保存当前渲染快照和日志到 `logs/`
- **ESC**: 退出并保存

## 📂 项目结构

- `include/`: 核心数学库与 AABB/BVH 定义
- `src/renderer_sycl.cpp`: SYCL 渲染核心 (GPU Kernels)
- `src/main.cpp`: 系统调度与 SDL2 显示逻辑
- `assets/`: 支持加载 `.obj` 3D 模型

---
*本项目作为从 CUDA 向 oneAPI 迁移的典型案例，展示了如何利用现代 C++ 释放移动端 SoC 的极致算力。*
