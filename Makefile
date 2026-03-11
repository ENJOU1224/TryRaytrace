# =============================================================================
# SYCL Raytracer Makefile (Intel oneAPI / Lunar Lake)
# =============================================================================

# 1. 目录结构
SRC_DIR   = src
INC_DIR   = include
BIN_DIR   = bin
PARALLEL_JOBS ?= $(shell nproc)

# 2. 编译器配置 (使用 Intel oneAPI icpx)
CXX = icpx
OV_ROOT = $(shell if [ -n "$$OpenVINO_DIR" ]; then cd "$$OpenVINO_DIR/../.." && pwd; fi)
OV_INCLUDE = -isystem $(OV_ROOT)/runtime/include
OV_LIBS = -L$(OV_ROOT)/runtime/lib/intel64 -Wl,-rpath,$(OV_ROOT)/runtime/lib/intel64 -lopenvino
# 下面这几项都是本机默认安装路径，目的是让日常使用尽量少敲长参数。
ONEAPI_COMPILER_LIB = $(HOME)/intel/oneapi/compiler/2025.3/lib
ONEAPI_COMPILER_OPT_LIB = $(HOME)/intel/oneapi/compiler/2025.3/opt/compiler/lib
OPENVINO_RUNTIME_LIB = $(HOME)/intel/openvino_2026/runtime/lib/intel64
# 本机默认 Embree 安装位置
EMBREE_ROOT ?= $(HOME)/intel/embree-4.4.0.sycl.x86_64.linux
EMBREE_INCLUDE = -isystem $(EMBREE_ROOT)/include
EMBREE_LIBS = $(EMBREE_ROOT)/lib/libembree4_sycl.a $(EMBREE_ROOT)/lib/libembree4.so -Wl,-rpath,$(EMBREE_ROOT)/lib -lur_loader
# 本机默认 Level Zero RTAS 支持库位置
RTAS_SUPPORT_ROOT ?= $(HOME)/Downloads/level-zero-raytracing-support-1.2.3/build
# 运行时库顺序非常重要：
# 1. 先走 oneAPI 编译器自己的 libsycl / libur_loader / libiomp5
# 2. 再走你本地编出来的 RTAS builder
# 3. 再走 OpenVINO 和 Embree
#
# 如果把 Embree 包里自带的 libsycl 放在前面，容易和 oneAPI 主程序运行时混用，
# 最终表现成“程序能编，但启动就崩”。
RUN_LD_LIBRARY_PATH = $(ONEAPI_COMPILER_LIB):$(ONEAPI_COMPILER_OPT_LIB):$(RTAS_SUPPORT_ROOT):$(OPENVINO_RUNTIME_LIB):$(EMBREE_ROOT)/lib:$$LD_LIBRARY_PATH

# SYCL 专用编译选项
# -fsycl: 核心标志，启用 SYCL 编译器驱动
# -O3: 最高优化等级
# -march=native: 针对 Lunar Lake 架构优化
# -qopenmp: 支持主循环中的图像后处理并行
CXXFLAGS = -O3 -fsycl -march=native -qopenmp \
           -I$(SRC_DIR) -I$(INC_DIR) $(OV_INCLUDE) -Wall -Wextra -MMD -MP \
           -D_REENTRANT

# 链接选项
LDFLAGS = -lSDL2 -fsycl -qopenmp $(OV_LIBS)

# 几何查询后端选择
# 当前默认值改为 embree。
# 如需切回旧的软件 BVH 查询后端：
#   make RAY_QUERY_BACKEND=software
RAY_QUERY_BACKEND ?= embree

ifeq ($(RAY_QUERY_BACKEND),software)
  CXXFLAGS += -DRAY_QUERY_BACKEND_SOFTWARE
else ifeq ($(RAY_QUERY_BACKEND),embree)
  CXXFLAGS += -DRAY_QUERY_BACKEND_EMBREE $(EMBREE_INCLUDE)
  LDFLAGS += $(EMBREE_LIBS)
else
  $(error Unsupported RAY_QUERY_BACKEND '$(RAY_QUERY_BACKEND)'; expected software or embree)
endif

# 用后端名字隔离目标文件目录，避免 software / embree 互相污染增量构建结果。
OBJ_DIR = $(BIN_DIR)/obj_$(RAY_QUERY_BACKEND)

# 3. 目标文件
TARGET = $(BIN_DIR)/sycl_engine
CHECK_NPU_TARGET = $(BIN_DIR)/check_npu
DEMO_TARGET = $(BIN_DIR)/denoise_demo
EMBREE_PROBE_TARGET = $(BIN_DIR)/embree_probe

# 源文件列表 (移除 renderer.cu 和 pipeline.cpp，因为我们简化了逻辑)
OBJS_NAMES = bvh.o renderer_sycl.o camera.o scene.o loader.o input.o image_io.o frame_processing.o perf_monitor.o denoiser_openvino.o async_denoiser.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))
DEMO_OBJS_NAMES = bvh.o renderer_sycl.o camera.o scene.o loader.o frame_processing.o denoiser_openvino.o denoise_demo.o
DEMO_OBJS = $(addprefix $(OBJ_DIR)/, $(DEMO_OBJS_NAMES))

# 4. 构建规则
all: dir $(TARGET)

dir:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

$(TARGET): $(OBJS)
	@echo "🔗 链接主程序: $@"
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(DEMO_TARGET): $(DEMO_OBJS)
	@echo "🔗 链接演示程序: $@"
	@$(CXX) $(DEMO_OBJS) -o $@ $(LDFLAGS)

$(EMBREE_PROBE_TARGET): $(SRC_DIR)/embree_probe.cpp | check-embree-env dir
	@echo "🔨 编译 $<"
	@$(CXX) -std=c++17 -O2 -fsycl -Wall -Wextra $< -o $@ \
	-I$(SRC_DIR) -I$(INC_DIR) $(EMBREE_INCLUDE) $(EMBREE_LIBS)

# 通用 C++ 编译规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | check-env dir
	@echo "🔨 编译 $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

run: all check-env
	@echo "🚀 运行 SYCL 光追引擎..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(TARGET)

run-no-denoise: all check-env
	@echo "🚀 运行 SYCL 光追引擎（关闭 NPU 降噪）..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" LUNAR_DISABLE_NPU_DENOISER=1 ./$(TARGET)

denoise-demo: $(DEMO_TARGET)
	@echo "✅ 演示程序已构建: $(DEMO_TARGET)"

run-denoise-demo: $(DEMO_TARGET) check-env
	@echo "🧪 运行单帧降噪对比演示..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(DEMO_TARGET)

embree-probe: $(EMBREE_PROBE_TARGET)
	@echo "✅ Embree SYCL 探针已构建: $(EMBREE_PROBE_TARGET)"

run-embree-probe: $(EMBREE_PROBE_TARGET) check-env
	@echo "🧪 运行 Embree SYCL 探针..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(EMBREE_PROBE_TARGET)

# ------------------------------------------------------------
# 便捷目标
# ------------------------------------------------------------
# 这些目标的设计原则是：
# - 构建阶段只“检测环境”，不主动 source
# - 运行阶段只补齐必要的 LD_LIBRARY_PATH
# - 常用路径都给默认值，日常尽量少打长命令
build-software:
	@$(MAKE) -j$(PARALLEL_JOBS) RAY_QUERY_BACKEND=software

build-embree:
	@$(MAKE) -j$(PARALLEL_JOBS) RAY_QUERY_BACKEND=embree EMBREE_ROOT="$(EMBREE_ROOT)"

run-embree: all check-env
	@echo "🚀 运行 Embree 后端..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(TARGET)

run-embree-no-denoise: all check-env
	@echo "🚀 运行 Embree 后端（关闭 NPU 降噪）..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" LUNAR_DISABLE_NPU_DENOISER=1 ./$(TARGET)

run-embree-probe-live: $(EMBREE_PROBE_TARGET) check-env
	@echo "🧪 运行 Embree SYCL 探针（注入 RTAS 支持库）..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(EMBREE_PROBE_TARGET)

check-env:
	@if [ -z "$$ONEAPI_ROOT" ] || ! command -v $(CXX) >/dev/null 2>&1; then \
		echo "错误：Intel oneAPI 环境未就绪。"; \
		echo "请先执行：source $$HOME/intel/oneapi/setvars.sh"; \
		exit 1; \
	fi
	@if [ -z "$$OpenVINO_DIR" ]; then \
		echo "错误：OpenVINO 环境未就绪。"; \
		echo "请先执行：source $$HOME/intel/openvino_2026/setupvars.sh"; \
		exit 1; \
	fi

check-embree-env:
	@if [ -z "$$ONEAPI_ROOT" ] || ! command -v $(CXX) >/dev/null 2>&1; then \
		echo "错误：Intel oneAPI 环境未就绪。"; \
		echo "请先执行：source $$HOME/intel/oneapi/setvars.sh"; \
		exit 1; \
	fi
	@if [ ! -f "$(EMBREE_ROOT)/include/embree4/rtcore.h" ]; then \
		echo "错误：未找到 Embree 头文件。"; \
		echo "当前 EMBREE_ROOT=$(EMBREE_ROOT)"; \
		exit 1; \
	fi
	@if [ ! -f "$(EMBREE_ROOT)/lib/libembree4.so" ] && [ ! -f "$(EMBREE_ROOT)/lib/libembree4.so.4" ]; then \
		echo "错误：未找到 Embree 库文件。"; \
		echo "当前 EMBREE_ROOT=$(EMBREE_ROOT)"; \
		exit 1; \
	fi

$(CHECK_NPU_TARGET): $(SRC_DIR)/check_npu.cpp | check-env dir
	@echo "🔨 编译 $<"
	@$(CXX) -std=c++17 -Wall -Wextra $< -o $@ \
	$(OV_INCLUDE) $(OV_LIBS)

checknpu: $(CHECK_NPU_TARGET) check-env
	@echo "🧪 检查 OpenVINO 设备..."
	@LD_LIBRARY_PATH="$(RUN_LD_LIBRARY_PATH)" ./$(CHECK_NPU_TARGET)

clean:
	@echo "🧹 清理构建产物..."
	@rm -rf $(BIN_DIR)

.PHONY: all dir clean run run-no-denoise check-env check-embree-env checknpu denoise-demo run-denoise-demo embree-probe run-embree-probe build-software build-embree run-embree run-embree-no-denoise run-embree-probe-live

-include $(OBJS:.o=.d) $(DEMO_OBJS:.o=.d)
