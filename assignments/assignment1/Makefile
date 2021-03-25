# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -std=c++17 $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
NVCCFLAGS	= -g -G --use_fast_math -arch=sm_35

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

all: matrix

OBJECTS 	= cpu_funcs.o gpu_funcs.o 
TARGET 		= matrix 

matrix: matrix.cu $(OBJECTS) matrix.h
	$(NVCC) -o $@ $< $(OBJECTS) $(NVCCFLAGS)

cpu_funcs.o: cpu_funcs.cu
	$(NVCC) -c $@ $< $(NVCCFLAGS)

gpu_funcs.o: gpu_funcs.cu
	$(NVCC) -c $@ $< $(NVCCFLAGS)

.PHONY:
	clean test run

clean:
	$(DEL_FILE) $(OBJECTS) $(TARGET)

test: matrix
	./test.sh

run: matrix
	./matrix -t -n 1000 -m 1000
	
