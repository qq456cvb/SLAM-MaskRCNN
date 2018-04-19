NVCC_FLAGS = -std=c++11
CUDA_OBJS = tsdf.o utils.o viewer.o

main: kernel.cpp link.o
	g++ -o main $(CUDA_OBJS) link.o ../kernel.cpp -lcudart -I/usr/local/cuda/include -std=c++11 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d

tsdf.o: ../tsdf.cu
	nvcc $(NVCC_FLAGS) -dc -I.. ../tsdf.cu

utils.o: ../utils.cu
	nvcc $(NVCC_FLAGS) -dc -I.. ../utils.cu

viewer.o: ../viewer.cu
	nvcc $(NVCC_FLAGS) -dc -I.. ../viewer.cu

kernel.cpp: ../kernel.cpp

link.o: $(CUDA_OBJS)
	nvcc -dlink $(CUDA_OBJS) -o link.o

all: main

clean: 
	rm *.o main

