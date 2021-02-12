CC=	g++
NVCC= 	nvcc

#Flags
#PARALLEL	= -fopenmp
#DEFINES	= -DWITH_OPENMP
CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
NVCCFLAGS	= -g -G --use_fast_math

###### Files
SOURCES		= addVectors.cu
OBJECTS		= addVectors.o

TARGET		= addVectors

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET)

addVectors.o: addVectors.cu
	$(NVCC) $< -c $(NVCCFLAGS)

clean:
	rm -f $(OBJECTS) $(TARGET)
