# =============================================================================
# High-Performance CUDA Raytracer Makefile
# =============================================================================
# 优化目标: 
# 1. 自动化: 自动处理头文件依赖关系 (修改 .h 自动重编相关 .cpp/.cu)
# 2. 高性能: 开启 AVX2 (CPU) 和 Fast Math (GPU)
# 3. 结构化: 源码/头文件/二进制/资源 分离
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 目录结构配置
# -----------------------------------------------------------------------------
SRC_DIR   = src
INC_DIR   = include
OBJ_DIR   = bin
ASSET_DIR = assets

# 最终可执行文件路径
TARGET = $(OBJ_DIR)/cuda_engine

# 源文件列表
# 这里我们显式列出对象文件，Make 会根据模式规则去 src/ 找对应的源文件
OBJS_NAMES = bvh.o gpu_context.o pipeline.o camera.o scene.o loader.o input.o image_io.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))

# -----------------------------------------------------------------------------
# 2. 编译器与选项配置
# -----------------------------------------------------------------------------

# --- Host Compiler (C++) ---
CXX = g++

# CXXFLAGS 详解:
# -O3             : 最高级别优化 (循环展开, 内联等)
# -march=native   : [性能关键] 针对本机 CPU (i7-9750H) 开启 AVX2/FMA 指令集
# -fopenmp        : 开启多线程支持
# -Wall -Wextra   : 开启大部分警告
# -Wno-unknown-pragmas : 忽略未知的 #pragma (防止 OpenMP 在非 MP 模式下报警)
# -I...           : 头文件搜索路径
# -MMD -MP        : [工程关键] 自动生成 .d 依赖文件，修改 .h 能自动重编
CXXFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Wno-unknown-pragmas \
           -I$(SRC_DIR) -I$(INC_DIR) -MMD -MP

# --- Device Compiler (CUDA) ---
NVCC = nvcc

# 架构设置: 1660 Ti 属于 Turing 架构，计算能力 7.5
ARCH = -arch=sm_75

# NVCC_FLAGS 详解:
# -O3             : Host 端代码优化
# -Xptxas -O3     : [性能关键] 告诉 PTX 汇编器进行最高级别优化
# --use_fast_math : [性能关键] 使用硬件内置的快速数学函数 (如 __sinf), 牺牲微小精度换取速度
# -I...           : 头文件搜索路径
NVCC_FLAGS = -O3 $(ARCH) --use_fast_math -Xptxas -O3 -I$(SRC_DIR) -I$(INC_DIR)

# --- Linker (链接器) ---
# 最终链接通常交给 NVCC 处理，它会自动传递参数给 GCC
LDFLAGS = -lSDL2 -lgomp -Xcompiler -fopenmp

# CUDA 头文件路径 (备用，防止 g++ 找不到 cuda_runtime.h)
CUDA_INC = -I/usr/local/cuda/include -I/usr/lib/nvidia-cuda-toolkit/include

# -----------------------------------------------------------------------------
# 3. 构建规则
# -----------------------------------------------------------------------------

# 默认目标
all: dir $(TARGET)

# 创建输出目录
dir:
	@mkdir -p $(OBJ_DIR)

# 链接步骤
# $^ 代表所有依赖文件 (即 $(OBJS))
# $@ 代表目标文件 (即 $(TARGET))
$(TARGET): $(OBJS)
	@echo "🔗 Linking $@"
	@$(NVCC) $^ -o $@ $(LDFLAGS)

# --- 规则: 编译 C++ 文件 (.cpp -> .o) ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "🔨 Compiling C++ $<"
	@$(CXX) $(CXXFLAGS) $(CUDA_INC) -c $< -o $@

# --- 规则: 编译 CUDA 文件 (.cu -> .o) ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "⚡ Compiling CUDA $<"
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# --- 引入自动生成的依赖文件 ---
# 如果 .d 文件存在，Make 会读取它，从而知道哪些 .cpp 依赖哪些 .h
-include $(OBJS:.o=.d)

# 运行程序
run: all
	@echo "🚀 Running..."
	@./$(TARGET)

# 清理构建产物
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf $(OBJ_DIR)

# 伪目标 (防止目录下有同名文件导致冲突)
.PHONY: all dir clean run
