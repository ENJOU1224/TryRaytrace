# =============================================================================
# Modern High-Performance CUDA Makefile
# =============================================================================

# 1. 编译器设置
# 使用系统默认编译器 (g++)，开启针对本机 CPU 的所有优化指令集 (AVX2/FMA等)
CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -Wall -Wno-unknown-pragmas

# NVCC 设置 (不再需要 -ccbin g++-11)
NVCC = nvcc
# 你的显卡架构 (1660 Ti = Turing = sm_75)
ARCH = -arch=sm_75
# 如果还是偶发头文件冲突，保留这个宏作为保险；如果不用也能跑，可以删掉 COMPAT_FLAGS
COMPAT_FLAGS = -D_AMXTILEINTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AVX512BF16VLINTRIN_H_INCLUDED -D_AVX512BF16INTRIN_H_INCLUDED
NVCC_FLAGS = -O3 $(ARCH) --use_fast_math $(COMPAT_FLAGS)

# 2. 路径设置
# 告诉 g++ 去哪里找 CUDA 头文件 (根据你的系统调整，通常在 /usr/local/cuda/include)
CUDA_INC = -I/usr/local/cuda/include -I/usr/lib/nvidia-cuda-toolkit/include

# 3. 链接设置
LDFLAGS = -lSDL2 -lgomp -Xcompiler -fopenmp

# 4. 文件列表
TARGET = cuda_engine
OBJS = renderer.o pipeline.o camera.o scene.o loader.o input.o image_io.o main.o

# =============================================================================
# 构建规则
# =============================================================================

all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# 编译 C++ 源代码 (.cpp) -> Host Code
# 使用 CXX (g++) 编译，享受 AVX2 加速
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c $< -o $@

# 编译 CUDA 源代码 (.cu) -> Device Code
# 使用 NVCC 编译
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(TARGET) *.o

run:all
	./cuda_engine

# 伪目标
.PHONY: all clean run
