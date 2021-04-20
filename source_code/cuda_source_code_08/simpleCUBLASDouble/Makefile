# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
PARALLEL	= -lcudart -lcublas
#DEFINES		= -DWITH_OPENMP

#CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
CFLAGS		= -m64 -O4 $(DEFINES)
#CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -m64 -O4 $(DEFINES)

#NVCCFLAGS	= -g -G --use_fast_math -lcurand

LDFLAGS		= -lgslcblas -lm -ffast-math

LIBS		= $(PARALLEL) -L/usr/local/cuda-9.1/lib64
INCPATH		= -I/usr/include/ -I/usr/local/cuda-9.1/include -I. -I.. -I../../common/inc
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= simpleCUBLASDouble.cpp
OBJECTS		= simpleCUBLASDouble.o

TARGET 		= simpleCUBLASDouble


all: main

main: $(OBJECTS)
	$(CC) $(CXXFLAGS) -o simpleCUBLASDouble $(OBJECTS) $(LIBS) -I$(INCPATH) $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CXXFLAGS) -c $(INCPATH) $<

clean:
	-$(DEL_FILE) $(OBJECTS)  $(TARGET)
