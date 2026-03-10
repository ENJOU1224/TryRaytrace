# =============================================================================
# SYCL Raytracer Makefile (Intel oneAPI / Lunar Lake)
# =============================================================================

# 1. 目录结构
SRC_DIR   = src
INC_DIR   = include
OBJ_DIR   = bin

# 2. 编译器配置 (使用 Intel oneAPI icpx)
CXX = icpx

# SYCL 专用编译选项
# -fsycl: 核心标志，启用 SYCL 编译器驱动
# -O3: 最高优化等级
# -march=native: 针对 Lunar Lake 架构优化
# -qopenmp: 支持主循环中的图像后处理并行
CXXFLAGS = -O3 -fsycl -march=native -qopenmp \
           -I$(SRC_DIR) -I$(INC_DIR) -Wall -Wextra \
           -D_REENTRANT

# 链接选项
LDFLAGS = -lSDL2 -fsycl -qopenmp

# 3. 目标文件
TARGET = $(OBJ_DIR)/sycl_engine

# 源文件列表 (移除 renderer.cu 和 pipeline.cpp，因为我们简化了逻辑)
OBJS_NAMES = bvh.o renderer_sycl.o camera.o scene.o loader.o input.o image_io.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))

# 4. 构建规则
all: dir $(TARGET)

dir:
	@mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	@echo "🔗 Linking SYCL Engine: $@"
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# 通用 C++ 编译规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "🔨 Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	@echo "🚀 Running SYCL Raytracer..."
	@./$(TARGET)

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf $(OBJ_DIR)

.PHONY: all dir clean run
