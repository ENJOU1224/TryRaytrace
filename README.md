# Lunar-Raytrace: 高性能 SYCL 光线追踪引擎

这是一个针对 **Intel Lunar Lake (Core Ultra 258V)** 架构深度优化的路径追踪渲染器。它成功地将原本的 NVIDIA CUDA 后端迁移到了现代的 **Intel oneAPI (SYCL)** 架构，实现了在集成核显 (Arc 140V) 上的高性能硬件加速。

## 🚀 核心特性

- **跨架构支持**: 基于 SYCL 标准编写，可在 Intel GPU (Level Zero/OpenCL) 和 CPU 上无缝切换。
- **Lunar Lake 优化**:
  - **USM (Unified Shared Memory)**: 利用 SoC 统一内存特性，实现 CPU/GPU 零拷贝数据传输。
  - **Xe2 矢量优化**: 关键路径使用 `sycl::native` 指令集和 FMA (乘加) 指令。
  - **SIMD 调度**: 当前主渲染使用针对本机实测调整后的工作组布局 `8x8`。
- **物理渲染算法**:
  - **BVH 加速结构**: 线性化 BVH 树，支持高速射线求交。
  - **NEE (Next Event Estimation)**: 显式光源采样，大幅降低阴影区域噪点。
  - **Pro 级 PBR 材质**: 支持金属 (Specular)、玻璃 (Refraction) 和漫反射 (Diffuse)，基于 Schlick 菲涅尔近似。
- **稳定性工程**:
  - **Firefly Clamping**: 分层处理直接光、命中光源、自发光与最终颜色，避免少量离群样本污染全图。
  - **鲁棒性防火墙**: 内置 NaN/Inf 过滤与负值清理。
  - **实时反馈**: 实时 FPS 统计、终端渲染进度刷新及自动 Snapshot/Log 保存。
- **NPU 帧降噪接口**:
  - 基于 OpenVINO C++ Runtime，在程序内直接构建轻量卷积降噪图并编译到 `NPU`。
  - 当前内置模型是一个 `5x5` 高斯残差卷积降噪器，带简单的高亮保护，输入输出均为当前窗口分辨率。
  - 主程序默认启用异步 NPU 流水线：GPU 持续渲染，NPU 后台处理最近提交的一帧，并显示最近完成的降噪结果。
  - 若未检测到 `NPU` 或 NPU 编译失败，渲染器会自动回退到原始画面，不影响主渲染流程。

## 🧪 已落地的专项优化

当前项目不是“原始 CUDA 示例直接迁移”的状态，而是已经做过几轮面向这台机器的专项优化：

1. **墙面材质分支修正**
   - 对高粗糙、非金属、非透射材质，直接按纯漫反射处理。
   - 这样可以避免墙面错误走入低概率 `specular` 路径，显著减少全局高亮离群样本。

2. **BVH 构建质量优化**
   - BVH 构建已经从简单中位数分裂升级为分桶 `SAH` 划分。
   - 同时把叶子阈值放宽到小批量三角形，减少树深和栈操作。

3. **三角形几何预计算**
   - CPU 端预烘焙 `edge1 / edge2 / normal / area`。
   - GPU kernel 不再反复现算这些静态几何量。
   - 当前对象布局是 `v0 + edge1 + edge2 + normal + area`，兼顾了性能和显存占用。

4. **路径成本控制**
   - 当前最大路径深度是 `5`
   - 俄罗斯轮盘赌从 `depth >= 3` 开始
   - `NEE` 当前只在 `depth < 3` 的浅层路径上启用

5. **显示链降本**
   - 关闭降噪时，主循环走“直接从累积缓冲转 ARGB”的快速路径
   - 静止观察时默认隔帧提交显示，减少 CPU 后处理和 SDL 提交开销

6. **异步 NPU 流水线**
   - GPU 渲染和 NPU 降噪解耦
   - 接受一定显示延迟，不在相机移动时强制作废旧帧
   - 标题栏会显示当前 `Lag`

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
   默认行为:
   - 启用异步 NPU 降噪流水线
   - 终端显示 `FPS / Frame / Present / DenoiseLag`
   - 自动记录 `logs/perf_*.csv` 性能日志

4. **检查 OpenVINO 是否识别到 NPU**:
   ```bash
   make checknpu
   ```

5. **运行单帧降噪对比演示**:
   ```bash
   make denoise-demo
   ./bin/denoise_demo
   ```
   演示程序会只渲染一帧，然后每 2 秒在“原始单帧图 / NPU 降噪图”之间自动切换。

## 🧠 NPU 降噪使用方式

当前版本不需要额外模型文件，也不依赖环境变量开关。运行逻辑固定如下：

1. 程序启动时先用 OpenVINO 查询当前是否存在 `NPU` 设备。
2. 如果存在，就在内存中构建一个轻量残差卷积降噪图，并编译到 `NPU`。
3. 主线程每次准备显示时，会把最近一帧线性 `RGB` 提交给后台降噪线程。
4. 后台线程持续处理“最近待处理帧”，并发布“最近完成的降噪结果”。
5. 显示端不等待 `NPU`，而是直接显示最近完成的一帧，允许一定显示延迟。
6. 如果没有检测到 `NPU`，或编译 / 推理失败，就直接显示原始画面。

内置模型说明：

- 模型结构是 `input -> 5x5 Gaussian Conv -> 高亮保护 -> 5x5 Gaussian Conv -> residual blend -> output`
- 残差融合比例固定为 `0.82 * 保护后原图 + 0.18 * 平滑结果`
- 这个模型不是追求极限画质的 SOTA 网络，而是为了当前项目阶段选择的稳定工程基线：
  - 不依赖外部模型文件
  - 算子非常简单，NPU 兼容性高
  - 代码可直接阅读，适合学习 OpenVINO 图构建与 NPU 部署

## 🔍 诊断模式

默认运行时不会打印逐帧诊断统计。
如果后续还需要继续排查 fireflies 或继续做性能实验，可以直接改下面几个代码开关：

- [main.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/main.cpp#L41) 的 `kEnableNpuDenoiser`
- [main.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/main.cpp#L42) 的 `kEnableDiagnosticStats`
- [main.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/main.cpp#L46) 和 [main.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/main.cpp#L47) 的工作组配置
- [renderer_sycl.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/renderer_sycl.cpp#L34) 到 [renderer_sycl.cpp](/home/enjou/temp/2026/3/TryRaytrace/src/renderer_sycl.cpp#L36) 的路径深度和 NEE 范围

`perf_*.csv` 当前会记录：
- `input_ms / render_ms / post_ms / present_ms / denoise_ms / total_ms / fps`
- `GPU` 两个 GT 的活动频率与估算忙碌率
- `NPU` 忙碌率、频率和内存占用
- 在开启诊断统计时，还会记录分项 clamp 计数

## 🎮 操作快捷键

- **WASD**: 移动相机
- **鼠标**: 旋转视角
- **P 键**: 保存当前渲染快照和日志到 `logs/`
- **ESC**: 退出并保存

## 📂 项目结构

- `include/`: 核心数学库与 AABB/BVH 定义
- `src/main.cpp`: 主调度器，负责输入、渲染、显示、异步降噪协同
- `src/renderer_sycl.cpp`: SYCL 渲染核心 (GPU Kernels)
- `src/async_denoiser.cpp`: 异步 NPU 降噪流水线
- `src/denoiser_openvino.cpp`: 内置 OpenVINO/NPU 降噪图
- `src/perf_monitor.cpp`: 性能日志与 GPU/NPU 代理指标采样
- `src/frame_processing.cpp`: 线性 RGB / 显示图像转换辅助函数
- `src/bvh.cpp`: BVH 构建，当前使用分桶 SAH 划分
- `assets/`: 支持加载 `.obj` 3D 模型

---
*本项目作为从 CUDA 向 oneAPI 迁移的典型案例，展示了如何利用现代 C++ 释放移动端 SoC 的极致算力。*
