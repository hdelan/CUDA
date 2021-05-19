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

bool verbose,timing,cpu,gpu;
int maxIterations;
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use

int parseArguments (int argc, char *argv[]);
void printUsage ();
void diff_matrices(float *A, float *C, unsigned int n, unsigned int m);
void diff_matrices(double *A, double *C, unsigned int n, unsigned int m);
void print_matrix_CPU(double * A, const unsigned int N, const unsigned int M);
double exponentialIntegralDouble (const int n,const double x);
float exponentialIntegralFloat (const int n,const float x);

#endif
