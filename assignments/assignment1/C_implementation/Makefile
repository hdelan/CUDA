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
NVCCFLAGS	= -g -G --use_fast_math

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

OBJECTS = matrix_funcs.o
TARGETS = matrix

all: matrix

matrix: matrix.cu matrix_funcs.o matrix.h
	$(NVCC) -o $@ matrix.cu matrix_funcs.o 

matrix_funcs.o: matrix_funcs.cu
	$(NVCC) -c $@ $< 


clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
