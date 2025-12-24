NVCC = nvcc
ARCH = -arch=sm_75
COMPAT_FLAGS = -D_AMXTILEINTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AVX512BF16VLINTRIN_H_INCLUDED -D_AVX512BF16INTRIN_H_INCLUDED
NVCC_FLAGS = -O3 $(ARCH) --use_fast_math $(COMPAT_FLAGS)
LDFLAGS = -lSDL2 -lgomp -Xcompiler -fopenmp

TARGET = cuda_engine
# 所有的 .o 文件
OBJS = renderer.o pipeline.o camera.o scene.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -ccbin g++-11 $(OBJS) -o $(TARGET) $(LDFLAGS)

# 通用规则：编译 .cpp -> .o
# main.cpp, camera.cpp, pipeline.cpp, scene.cpp 都可以用这个规则
%.o: %.cpp
	$(NVCC) -ccbin g++-11 $(NVCC_FLAGS) -Xcompiler -fopenmp -c $< -o $@

# 特殊规则：编译 .cu -> .o
renderer.o: renderer.cu renderer.h common.h scene.h
	$(NVCC) -ccbin g++-11 $(NVCC_FLAGS) -c renderer.cu -o renderer.o

run: all
	./cuda_engine

clean:
	rm -f $(TARGET) *.o
