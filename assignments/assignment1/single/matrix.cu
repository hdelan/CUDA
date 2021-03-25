/**
 * \file:        matrix.cu
 * \brief:       The main function for assignment 1
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-03-25
 */
#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"


int main(int argc, char * argv[]) {

  /*                     INITIALIZING                       */
  
  // Default values for N, M, block_size
  unsigned int N {10}, M {10};
  unsigned int block_size = 32;

  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};

  // Lots of time variables to keep track of time
  struct timeval start_time, cpu_time_rowsum, cpu_time_colsum, cpu_time_reduction, gpu_time_rowsum, gpu_time_colsum, gpu_time_reduction, cpu_time_rowsum1, cpu_time_colsum1, cpu_time_reduction1, gpu_time_rowsum1, gpu_time_colsum1, gpu_time_reduction1;
 
  // Default seed
  long unsigned int seed {1234};
  
  /*              GET OPTIONAL PARAMETERS                 */
  parse_command_line(argc, argv, N, M, seed, start_time, print_time, block_size);
  
  // Seed RNG
  srand48(seed);
  
  // Populate matrix with values from [-10.0, 10.0]
  float * A {(float *) malloc(sizeof(float) * N * M)};
  for (unsigned int i = 0; i < N*M; i++)
    A[i] = (float) drand48()*20.0 - 10.0;  

  float * rowsum {(float *) malloc(sizeof(float) * N)};
  float * colsum {(float *) malloc(sizeof(float) * M)};

  /*                     CPU STUFF                       */
  
  // Print options
  /*std::cout << std::setw(64) << std::setfill('~') << '\n';
  std::cout << "\t\t\tCPU\n";
  std::cout << std::setw(64) << std::setfill('~') << '\n' << std::setfill(' ');
  print_matrix_CPU(A, N, M);
*/

  /* CPU ROWSUM */ 
  gettimeofday(&cpu_time_rowsum, NULL);
  sum_abs_rows_CPU(A, rowsum, N, M);
  gettimeofday(&cpu_time_rowsum1, NULL);
  
  /* CPU COLSUM */ 
  gettimeofday(&cpu_time_colsum, NULL);
  sum_abs_cols_CPU(A, colsum, N, M);
  gettimeofday(&cpu_time_colsum1, NULL);

  /* CPU REDUCTION */ 
  gettimeofday(&cpu_time_reduction, NULL);
  float sum_of_rowsums_CPU = vector_reduction_CPU(rowsum, N);
  float sum_of_colsums_CPU = vector_reduction_CPU(colsum, M);
  gettimeofday(&cpu_time_reduction1, NULL);
  
  // Option to print rowsums & colsums
  //std::cout << "Rowsums: \n";
  //print_matrix_CPU(rowsum, N, 1);
  //std::cout << "Column sums: \n";
  //print_matrix_CPU(colsum, 1, M);

  //std::cout << "Sum of rowsums: " << std::setprecision(20) << sum_of_rowsums_CPU << std::endl;
  //std::cout << "Sum of colsums: " << sum_of_colsums_CPU << '\n' << std::endl;
  
  /* Calculating CPU times */
  double rowsum_time_CPU = ((double) cpu_time_rowsum1.tv_sec - cpu_time_rowsum.tv_sec) + (((double)(cpu_time_rowsum1.tv_usec - cpu_time_rowsum.tv_usec))/1000000.0);
  double colsum_time_CPU = ((double) cpu_time_colsum1.tv_sec - cpu_time_colsum.tv_sec) + (((double)(cpu_time_colsum1.tv_usec - cpu_time_colsum.tv_usec))/1000000.0);
  double reduction_time_CPU = ((double) cpu_time_reduction1.tv_sec - cpu_time_reduction.tv_sec) + (((double)(cpu_time_reduction1.tv_usec - cpu_time_reduction.tv_usec))/1000000.0);


  /*                     GPU STUFF                       */

  //std::cout << std::setw(64) << std::setfill('~') << '\n';
  //std::cout << "\t\t\tGPU\n";
  //std::cout << std::setw(64) << std::setfill('~') << '\n' << std::setfill(' ');

  
  /*                     INITIALIZING                       */
  float * A_d, * rowsum_d, * colsum_d;

  cudaMalloc((void **) &A_d, sizeof(float)*N*M);
  cudaMalloc((void **) &rowsum_d, sizeof(float)*N);
  cudaMalloc((void **) &colsum_d, sizeof(float)*M);

  cudaMemcpy(A_d, A, sizeof(float)*N*M, cudaMemcpyHostToDevice);

  // Initialize block_size
  dim3 dimBlock {block_size};


  /* GPU ROWSUM */
  dim3 dimGrid_row {(N/dimBlock.x) + (!(N%dimBlock.x)?0:1)};
  gettimeofday(&gpu_time_rowsum, NULL);
  sum_abs_rows_GPU<<<dimGrid_row, dimBlock>>>(A_d, rowsum_d, N, M);
  gettimeofday(&gpu_time_rowsum1, NULL);

  /* GPU COLSUM */
  dim3 dimGrid_col {(M/dimBlock.x) + (!(M%dimBlock.x)?0:1)};
  gettimeofday(&gpu_time_colsum, NULL);
  sum_abs_cols_GPU<<<dimGrid_col, dimBlock>>>(A_d, colsum_d, N, M);
  gettimeofday(&gpu_time_colsum1, NULL);
  
  /* GPU REDUCTIONS */
  gettimeofday(&gpu_time_reduction, NULL);
  float sum_of_rowsums_GPU = vector_reduction_GPU(rowsum_d, N, dimBlock, dimGrid_col);
  float sum_of_colsums_GPU = vector_reduction_GPU(colsum_d, M, dimBlock, dimGrid_row);
  gettimeofday(&gpu_time_reduction1, NULL);
  
  /* Option to print rowsums, colsums if N <= 100 or M <= 100 */
  //cudaMemcpy(rowsum, rowsum_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  //std::cout << "Rowsums: \n";
  //print_matrix_CPU(rowsum, N, 1);
  //cudaMemcpy(colsum, colsum_d, sizeof(float)*M, cudaMemcpyDeviceToHost);
  //std::cout << "Column sums: \n";
  //print_matrix_CPU(colsum, 1, M);
  
  //std::cout << "GPU Sum of rowsums: " << std::setprecision(20) << sum_of_rowsums_GPU << std::endl;
  //std::cout << "GPU Sum of colsums: " << std::setprecision(20) << sum_of_colsums_GPU << std::endl;
  
  
  /* Calculating GPU times */
  double rowsum_time_GPU = ((double) gpu_time_rowsum1.tv_sec - gpu_time_rowsum.tv_sec) + (((double)(gpu_time_rowsum1.tv_usec - gpu_time_rowsum.tv_usec))/1000000.0);
  double colsum_time_GPU = ((double) gpu_time_colsum1.tv_sec - gpu_time_colsum.tv_sec) + (((double)(gpu_time_colsum1.tv_usec - gpu_time_colsum.tv_usec))/1000000.0);
  double reduction_time_GPU = ((double) gpu_time_reduction1.tv_sec - gpu_time_reduction.tv_sec) + (((double)(gpu_time_reduction1.tv_usec - gpu_time_reduction.tv_usec))/1000000.0);
  
  /* PRINT OUTPUT */
  if (print_time == 1) {
  
	std::cout << std::setw(64) << std::setfill('~') << '\n';
	std::cout << "\t  CPU TIME    GPU TIME\t      SPEEDUP\t\t  ERROR\n";

	std::cout << std::setw(64) << std::setfill('~') << '\n' << std::setfill(' ');

	std::cout << "Rowsum" << std::setw(12) << std::setprecision(10) 
	  << rowsum_time_CPU << std::setw(12) 
	  << rowsum_time_GPU << std::setw(15) 
	  << rowsum_time_CPU/rowsum_time_GPU << std::setw(18) 
	  << std::fabs((sum_of_rowsums_GPU - sum_of_rowsums_CPU)/sum_of_rowsums_CPU) << std::endl;
	std::cout << "Colsum"  << std::setw(12) 
	  << colsum_time_CPU << std::setw(12) 
	  << colsum_time_GPU << std::setw(15) 
	  << colsum_time_CPU/colsum_time_GPU << std::setw(18) 
	  << std::fabs((sum_of_colsums_GPU - sum_of_colsums_CPU)/sum_of_colsums_CPU) << std::endl;

	std::cout << "Reduce" <<  std::setw(12)
	  << reduction_time_CPU <<std::setw(12) 
	  << reduction_time_GPU << std::setw(15) 
	  << reduction_time_CPU/reduction_time_GPU << std::setw(18) << std::endl;
  
}
  
  std::cout << '\n';
  
  return 0;
}

