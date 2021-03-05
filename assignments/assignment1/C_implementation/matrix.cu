#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"

int main(int argc, char * argv[]) {

  // Default values for N, m
  unsigned int N {10}, M {10};
  int i = 0;
  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};
  struct timeval start_time;
 
  // Default seed
  long unsigned int seed {123456};
  std::cout << i << std::endl;
  i++;
  // Get optional parameters
  parse_command_line(argc, argv, N, M, seed, start_time, print_time);
  
  // Seed RNG
  srand48(seed);
  
  // Populate matrix with values from [-10.0, 10.0]
  float * A {(float *) malloc(sizeof(float) * N * M)};
  for (unsigned int i = 0; i < N*M; i++)
    A[i] = (float) drand48()*20.0 - 10.0;  

  std::setfill('~');
  std::cout << std::setw(24) << "CPU" << 
  print_matrix_CPU(A, N, M);

  float * rowsum {(float *) malloc(sizeof(float) * N)};
  float * colsum {(float *) malloc(sizeof(float) * M)};

  sum_abs_rows_CPU(A, rowsum, N, M);
  print_matrix_CPU(rowsum, N, 1);
  
  sum_abs_cols_CPU(A, colsum, N, M);
  print_matrix_CPU(colsum, 1, M);
 
  std::cout << "Sum of rowsums: " << vector_reduction_CPU(rowsum, N) << std::endl;
  std::cout << "Sum of colsums: " << vector_reduction_CPU(colsum, M) << std::endl;
  return 0;
}

#ifndef MATRIX_H
#define MATRIX_H

// KERNELS
__global__ void sum_abs_rows_GPU(float * data, float * rowsum, int N, int M);
__global__ void sum_abs_cols_GPU(float * data, float * colsum, int N, int M);

// CPU MATRIX FUNCTIONS
float vector_reduction_CPU(const float * vector, const int n); 
void sum_abs_rows_CPU(float * matrix, float * rowsum, int N, int M);
void sum_abs_cols_CPU(float * matrix, float * colsum, int N, int M);

// HELPER FUNCTIONS
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, long unsigned int & seed, struct timeval & start_time, int & print_time);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

#endif

