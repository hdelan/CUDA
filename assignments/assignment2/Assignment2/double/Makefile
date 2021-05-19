# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

OPT = -O3

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -std=c++17 $(PARALLEL) $(DEFINES)
CXXFLAGS	= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall 
NVCCFLAGS	= --use_fast_math -arch=sm_35 -Wno-deprecated-gpu-targets -lineinfo #--ptxas-options=-v

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

all: radiator

OBJECTS 	= cpu_funcs.o gpu_funcs.o 
TARGET 		= radiator

radiator: main.cu $(OBJECTS) 
	$(NVCC) -o $@ $< $(OBJECTS) $(NVCCFLAGS)

cpu_funcs.o: cpu_funcs.cpp
	$(CXX) -c $< $(CXXFLAGS)

gpu_funcs.o: gpu_funcs.cu 
	$(NVCC) -c $< $(NVCCFLAGS)

.PHONY:
	clean test run prof

clean:
	$(DEL_FILE) $(OBJECTS) $(TARGET)

prof: radiator
	./radiator -c -n 15360 -m 15360 -p 500 -t -a

test: radiator
	./test.sh 
