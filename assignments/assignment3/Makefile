CC=g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#CFLAGS= -g -pedantic -W -Wall -L/usr/lib
CFLAGS		= -W -Wall -std=c++17 $(PARALLEL) $(DEFINES)
CXXFLAGS	= -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall 
NVCCFLAGS	= --use_fast_math -arch=sm_35 -Wno-deprecated-gpu-targets -lineinfo #--ptxas-options=-v

INCPATH       	= -I. -I.

TARGET		=expint
OBJECTS		=gpu_funcs.o cpu_funcs.o
EXEC		=exponentialIntegral.out

expint: main.cu gpu_funcs.o cpu_funcs.o
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

gpu_funcs.o: gpu_funcs.cu
	$(NVCC) -c $< $(NVCCFLAGS)

cpu_funcs.o: cpu_funcs.cpp
	$(CC) -c $< $(CXXFLAGS)

.PHONY:
	clean test run prof

clean:
	$(DEL_FILE) *.o $(TARGET)

prof: expint
	sudo nvprof ./expint -n 10000 -m 10000 -c

run: expint
	./expint
