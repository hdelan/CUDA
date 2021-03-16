#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"

int BLOCK_SIZE = 64;

int main(int argc, char * argv[]) {

  // Default values for N, m
  unsigned int N {10}, M {10};
  int i = 0;
  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};
  struct timeval start_time, cpu_time, gpu_time;
 
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

  std::cout << std::setw(60) << std::setfill('~') << '\n';
  std::cout << "\t\t\tCPU\n";
  std::cout << std::setw(60) << std::setfill('~') << '\n' << std::setfill(' ');
  print_matrix_CPU(A, N, M);

  float * rowsum {(float *) malloc(sizeof(float) * N)};
  float * colsum {(float *) malloc(sizeof(float) * M)};

  // CPU STUFF
  sum_abs_rows_CPU(A, rowsum, N, M);
  std::cout << "Rowsums: \n";
  print_matrix_CPU(rowsum, N, 1);
  
  sum_abs_cols_CPU(A, colsum, N, M);
  std::cout << "Column sums: \n";
  print_matrix_CPU(colsum, 1, M);
 
  std::cout << "Sum of rowsums: " << vector_reduction_CPU(rowsum, N) << std::endl;
  std::cout << "Sum of colsums: " << vector_reduction_CPU(colsum, M) << std::endl;
  
  std::cout << '\n';

  if (print_time == 1) {
	  gettimeofday(&cpu_time, NULL);
	  std::cout << "CPU time taken: " << ((double) cpu_time.tv_sec - start_time.tv_sec) + (((double)(cpu_time.tv_usec - start_time.tv_usec))/1000000.0) << "\n\n";
  }

  // GPU STUFF
  std::cout << std::setw(60) << std::setfill('~') << '\n';
  std::cout << "\t\t\tGPU\n";
  std::cout << std::setw(60) << std::setfill('~') << '\n' << std::setfill(' ');

  float * A_d, * rowsum_d, * colsum_d;

  cudaMalloc((void **) &A_d, sizeof(float)*N*M);
  cudaMalloc((void **) &rowsum_d, sizeof(float)*N);
  cudaMalloc((void **) &colsum_d, sizeof(float)*M);

  cudaMemcpy(A_d, A, sizeof(float)*N*M, cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((N/dimBlock.x) + (!(N%dimBlock.x)?0:1));


  sum_abs_rows_GPU<<<dimGrid, dimBlock>>>(A_d, rowsum_d, N, M);
  sum_abs_cols_GPU<<<dimGrid, dimBlock>>>(A_d, colsum_d, N, M);

  cudaMemcpy(rowsum, rowsum_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(colsum, colsum_d, sizeof(float)*M, cudaMemcpyDeviceToHost);
  
  std::cout << "Rowsums: \n";
  print_matrix_CPU(rowsum, N, 1);
  std::cout << "Sum of rowsums: " << vector_reduction_CPU(rowsum, N) << std::endl;
  std::cout << "Sum of rowsums: " << vector_reduction_GPU(rowsum, N, dimBlock, dimGrid) << std::endl;
  
  std::cout << "Column sums: \n";
  print_matrix_CPU(colsum, 1, M);
  std::cout << "Sum of colsums: " << vector_reduction_CPU(colsum, M) << std::endl;
  
  if (print_time == 1) {
	  gettimeofday(&gpu_time, NULL);
	  std::cout << "GPU time taken: " << ((double) gpu_time.tv_sec - cpu_time.tv_sec) + (((double)(gpu_time.tv_usec - cpu_time.tv_usec))/1000000.0) << "\n\n";
  }
  
  std::cout << '\n';
  
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

