# Makefile
# $< means first dependency of the current target
# $+ means all current dependencies
# $@ means current target

# Variables for the makefile
CC = gcc
#CFLAGS = -Wall -O2  -ffast-math
CFLAGS = -Wall -O2 -ffast-math
LDFLAGS = -lgslcblas -lm -ffast-math
#LDFLAGS = -lm -ffast-math -llapack -lblas -lgsl -lgslcblas -L/usr/lib/lapack/ -L/usr/lib/libblas/

INCPATH		= -I/usr/include/ -I/usr/lib/x86_64-linux-gnu/ -I. -I.. -I../../common/inc


# default target
blasTest.out: blasTest.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

# Those are not necessary
#%.o: %.c
#	$(CC) -c $< $(CFLAGS)

test: blasTest.out
	./blasTest.out

#clean
.PHONY: clean
clean:
	rm -f *.o *.out $(TARGET)
