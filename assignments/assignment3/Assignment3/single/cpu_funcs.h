#ifndef CPU_FUNCS_H_123
#define CPU_FUNCS_H_123

#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>

extern bool cpu, gpu, verbose, timing, split;
extern float a, b;
extern unsigned numberOfSamples, n, block_size;
extern int maxIterations;

void printUsage ();
void diff_matrices(float *A, float *C, unsigned int n, unsigned int m);
void diff_matrices(float *A, float *C, unsigned int n, unsigned int m);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);
float exponentialIntegralFloat (const int n,const float x);
float exponentialIntegralFloat (const int n,const float x);
int parseArguments (int argc, char *argv[]);

#endif
