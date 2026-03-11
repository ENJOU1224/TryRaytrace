# =============================================================================
# SYCL Raytracer Makefile (Intel oneAPI / Lunar Lake)
# =============================================================================

# 1. 目录结构
SRC_DIR   = src
INC_DIR   = include
OBJ_DIR   = bin

# 2. 编译器配置 (使用 Intel oneAPI icpx)
CXX = icpx
OV_ROOT = $(shell if [ -n "$$OpenVINO_DIR" ]; then cd "$$OpenVINO_DIR/../.." && pwd; fi)
OV_INCLUDE = -isystem $(OV_ROOT)/runtime/include
OV_LIBS = -L$(OV_ROOT)/runtime/lib/intel64 -Wl,-rpath,$(OV_ROOT)/runtime/lib/intel64 -lopenvino

# SYCL 专用编译选项
# -fsycl: 核心标志，启用 SYCL 编译器驱动
# -O3: 最高优化等级
# -march=native: 针对 Lunar Lake 架构优化
# -qopenmp: 支持主循环中的图像后处理并行
CXXFLAGS = -O3 -fsycl -march=native -qopenmp \
           -I$(SRC_DIR) -I$(INC_DIR) $(OV_INCLUDE) -Wall -Wextra \
           -D_REENTRANT

# 链接选项
LDFLAGS = -lSDL2 -fsycl -qopenmp $(OV_LIBS)

# 3. 目标文件
TARGET = $(OBJ_DIR)/sycl_engine
CHECK_NPU_TARGET = $(OBJ_DIR)/check_npu
DEMO_TARGET = $(OBJ_DIR)/denoise_demo

# 源文件列表 (移除 renderer.cu 和 pipeline.cpp，因为我们简化了逻辑)
OBJS_NAMES = bvh.o renderer_sycl.o camera.o scene.o loader.o input.o image_io.o denoiser_openvino.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))
DEMO_OBJS_NAMES = bvh.o renderer_sycl.o camera.o scene.o loader.o denoiser_openvino.o denoise_demo.o
DEMO_OBJS = $(addprefix $(OBJ_DIR)/, $(DEMO_OBJS_NAMES))

# 4. 构建规则
all: dir $(TARGET)

dir:
	@mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	@echo "🔗 链接主程序: $@"
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(DEMO_TARGET): $(DEMO_OBJS)
	@echo "🔗 链接演示程序: $@"
	@$(CXX) $(DEMO_OBJS) -o $@ $(LDFLAGS)

# 通用 C++ 编译规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | check-env dir
	@echo "🔨 编译 $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	@echo "🚀 运行 SYCL 光追引擎..."
	@./$(TARGET)

denoise-demo: $(DEMO_TARGET)
	@echo "✅ 演示程序已构建: $(DEMO_TARGET)"

run-denoise-demo: $(DEMO_TARGET)
	@echo "🧪 运行单帧降噪对比演示..."
	@./$(DEMO_TARGET)

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

$(CHECK_NPU_TARGET): $(SRC_DIR)/check_npu.cpp | check-env dir
	@echo "🔨 编译 $<"
	@$(CXX) -std=c++17 -Wall -Wextra $< -o $@ \
	$(OV_INCLUDE) $(OV_LIBS)

checknpu: $(CHECK_NPU_TARGET)
	@echo "🧪 检查 OpenVINO 设备..."
	@./$(CHECK_NPU_TARGET)

clean:
	@echo "🧹 清理构建产物..."
	@rm -rf $(OBJ_DIR)

.PHONY: all dir clean run check-env checknpu denoise-demo run-denoise-demo
