# =============================================================================
# Clean & Fast CUDA Makefile
# =============================================================================

# 1. 目录配置
SRC_DIR = src
INC_DIR = include
OBJ_DIR = bin
ASSET_DIR = assets

# 目标文件
TARGET = $(OBJ_DIR)/cuda_engine

# 源文件列表
OBJS_NAMES = renderer.o pipeline.o camera.o scene.o loader.o input.o image_io.o main.o
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))

# =============================================================================
# 编译器配置
# =============================================================================

# Host Compiler (C++)
# 使用系统默认 g++，开启 AVX2/FMA 加速
CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -Wall -Wno-unknown-pragmas -I$(SRC_DIR) -I$(INC_DIR)

# Device Compiler (CUDA)
# 使用系统默认 nvcc
NVCC = nvcc
ARCH = -arch=sm_75
NVCC_FLAGS = -O3 $(ARCH) --use_fast_math -I$(SRC_DIR) -I$(INC_DIR)

# 链接器
LDFLAGS = -lSDL2 -lgomp -Xcompiler -fopenmp

# CUDA 头文件路径 (如果报错找不到 cuda_runtime.h，请检查这里)
CUDA_INC = -I/usr/local/cuda/include -I/usr/lib/nvidia-cuda-toolkit/include

# =============================================================================
# 构建规则
# =============================================================================

all: dir $(TARGET)

dir:
	@mkdir -p $(OBJ_DIR)

# 链接
$(TARGET): $(OBJS)
	@echo "Linking $@"
	@$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# 编译 C++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++ $<"
	@$(CXX) $(CXXFLAGS) $(CUDA_INC) -c $< -o $@

# 编译 CUDA
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA $<"
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 运行
run: all
	./$(TARGET)

# 清理
clean:
	rm -rf $(OBJ_DIR)

.PHONY: all dir clean run
