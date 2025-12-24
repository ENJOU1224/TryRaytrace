# =============================================================================
# Project Configuration
# =============================================================================

# 目录定义
SRC_DIR = src
OBJ_DIR = bin
# 资源目录 (运行时需要，这里只用于标记)
ASSET_DIR = assets

# 目标文件
TARGET = $(OBJ_DIR)/cuda_engine

# 源文件列表 (手动列出，或者用 wildcard 通配符)
# 这里为了清晰，我们手动列出对象名，Make 会自动去 src 找对应的 cpp/cu
OBJS_NAMES = renderer.o pipeline.o camera.o scene.o loader.o input.o image_io.o main.o
# 给所有对象加上 bin/ 前缀
OBJS = $(addprefix $(OBJ_DIR)/, $(OBJS_NAMES))

# =============================================================================
# Compiler Settings
# =============================================================================

# Host Compiler
CXX = g++
# -I$(SRC_DIR): 告诉编译器头文件在 src 目录里
CXXFLAGS = -O3 -march=native -fopenmp -Wall -Wno-unknown-pragmas -I$(SRC_DIR)

# Device Compiler
NVCC = nvcc
ARCH = -arch=sm_75
COMPAT_FLAGS = -D_AMXTILEINTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AVX512BF16VLINTRIN_H_INCLUDED -D_AVX512BF16INTRIN_H_INCLUDED
# 同样需要 -I$(SRC_DIR)
NVCC_FLAGS = -O3 $(ARCH) --use_fast_math $(COMPAT_FLAGS) -I$(SRC_DIR)

# Linker Flags
LDFLAGS = -lSDL2 -lgomp -Xcompiler -fopenmp

# CUDA Include Path (根据你的环境调整)
CUDA_INC = -I/usr/local/cuda/include -I/usr/lib/nvidia-cuda-toolkit/include

# =============================================================================
# Build Rules
# =============================================================================

# 默认目标
all: dir $(TARGET)

# 创建 bin 目录
dir:
	@mkdir -p $(OBJ_DIR)

# 链接
$(TARGET): $(OBJS)
	@echo "Linking $@"
	@$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# 编译 C++ (.cpp -> .o)
# $< 是源文件 (src/xxx.cpp), $@ 是目标文件 (bin/xxx.o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++ $<"
	@$(CXX) $(CXXFLAGS) $(CUDA_INC) -c $< -o $@

# 编译 CUDA (.cu -> .o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA $<"
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 运行 (方便调试)
run: all
	./$(TARGET)

# 清理
clean:
	rm -rf $(OBJ_DIR)

.PHONY: all dir clean run
