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

OBJECTS 	= matrix_funcs.o
TARGET 		= matrix 

matrix: matrix.cu matrix_funcs.o matrix.h
	$(NVCC) -o $@ matrix.cu matrix_funcs.o 

matrix_funcs.o: matrix_funcs.cu
	$(NVCC) -c $@ $< 

.PHONY:
	clean test

clean:
	$(DEL_FILE) $(OBJECTS) $(TARGET)

test:
	echo "/*   N=M=1000   */\n\n" > output
	./matrix -n 1000 -m 1000 -t >> output
	echo "/*   N=M=5000   */\n\n" >> output
	./matrix -n 5000 -m 5000 -t >> output
	echo "/*   N=M=10000   */\n\n" >> output
	./matrix -n 10000 -m 10000 -t >> output
	echo "/*   N=M=25000   */\n\n" >> output
	./matrix -n 25000 -m 25000 -t >> output
	 
