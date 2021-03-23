#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"

int BLOCK_SIZE = 64;

int main(int argc, char * argv[]) {

  // Default values for N, M
  unsigned int N {10}, M {10};

  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};
  struct timeval start_time, cpu_time, gpu_time;
 
  // Default seed
  long unsigned int seed {123456};
  
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
 
  std::cout << "Sum of rowsums: " << std::setprecision(20) << vector_reduction_CPU(rowsum, N) << std::endl;
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
  dim3 dimGrid((std::max(N, M)/dimBlock.x) + (!(std::max(N,M)%dimBlock.x)?0:1));

  sum_abs_rows_GPU<<<dimGrid, dimBlock>>>(A_d, rowsum_d, N, M);
  sum_abs_cols_GPU<<<dimGrid, dimBlock>>>(A_d, colsum_d, N, M);

  cudaMemcpy(rowsum, rowsum_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(colsum, colsum_d, sizeof(float)*M, cudaMemcpyDeviceToHost);
  
  std::cout << "Rowsums: \n";
  print_matrix_CPU(rowsum, N, 1);
  std::cout << "CPU Sum of rowsums: " << std::setprecision(20) << vector_reduction_CPU(rowsum, N) << std::endl;
  std::cout << "GPU Sum of rowsums: " << std::setprecision(20) << vector_reduction_GPU(rowsum_d, N, dimBlock, dimGrid) << std::endl;
  
  std::cout << "Column sums: \n";
  print_matrix_CPU(colsum, 1, M);
  std::cout << "CPU Sum of colsums: " << std::setprecision(20)<< vector_reduction_CPU(colsum, M) << std::endl;
  std::cout << "GPU Sum of colsums: " << std::setprecision(20) << vector_reduction_GPU(colsum_d, M, dimBlock, dimGrid) << std::endl;
  
  if (print_time == 1) {
	  gettimeofday(&gpu_time, NULL);
	  std::cout << "GPU time taken: " << ((double) gpu_time.tv_sec - cpu_time.tv_sec) + (((double)(gpu_time.tv_usec - cpu_time.tv_usec))/1000000.0) << "\n\n";
  }
  
  std::cout << '\n';
  
  return 0;
}

