/**
 * \file:        main.cu
 * \brief:       A main function to calculate the cylindrical radiator finite differences
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-04-29
 */

#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <cuda_profiler_api.h> 

#include "cpu_funcs.hpp"
#include "gpu_funcs.h"

int main(int argc, char * argv[]) { 
  unsigned int n {15360}, m {15360}, block_size {32}, num_iters {500};
  long unsigned int seed {123};
  int print_time {0}, cpu_calc {1}, diff_mats {1}, write_mat {0}, avg {0};
  struct timeval t1, t2, t3, t4;
  parse_command_line(argc, argv, n, m, num_iters, seed, print_time, cpu_calc, block_size, write_mat, avg);

  std::cout << "n: " << n << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "iters: " << num_iters << "\n";
  std::cout << "print_time: " << print_time << "\n";
  std::cout << "block_size: " << block_size << "\n";
  // A 
  double * A = (double *) calloc(n*m, sizeof(double));
  double * C = (double *) calloc(n*m, sizeof(double));

  // Set boundary conditions
  for (auto i=0;i<n;++i) {
    A[i*m] = C[i*m] = 1.0f*(double)(i+1)/(double)n;
    A[i*m+1] = C[i*m+1] = 0.80f*(double)(i+1)/(double)n;
  }

  double *A_d, *avg_d;
  cudaMalloc((void **) &A_d, sizeof(double)*n*m);
  cudaMalloc((void **) &avg_d, sizeof(double)*n);
  cudaMemcpy(A_d, A, sizeof(double)*n*m, cudaMemcpyHostToDevice);

  // Initialize block_size
  dim3 threads {block_size};
  dim3 blocks {(n/threads.x) + (!(n%threads.x)?0:1)};
  dim3 n_blocks {n};

  std::cout << "Num blocks: " << n_blocks.x << '\n';


  /************ CPU FUNCTION/READ IN FILE *************/
  // This file will be written to if the -c flag is not provided, or it will be read from
  std::string sparse_filename {"data/sp_CPU" + std::to_string(n) + "x" + std::to_string(m) + "_p" + std::to_string(num_iters) + ".txt"};

  if (cpu_calc == 1) {
    gettimeofday(&t1, NULL);
    cpu_rad_sweep1(C, n, m, num_iters, A);
    gettimeofday(&t2, NULL);
    printf("Finised CPU operation. Time taken: %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    // Every time the CPU code runs it will save the matrix to file
    print_sparse_matrix_to_file(sparse_filename, C, n, m, num_iters);
  } else {
    try {
      std::cout << "Reading SPARSE CPU-generated comparison matrix from file.\n";
      gettimeofday(&t1, NULL);
      read_sparse_matrix_from_file(sparse_filename, C);
      // Uncomment this to read a non-sparse matrix from file
      //read_matrix_from_file(other_filename, C);
      gettimeofday(&t2, NULL);
      printf("Finished reading sparse matrix from file. Time taken: %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    } catch (std::exception e) {
      std::cout << "Could not read from file! Will not be able to compare GPU matrix with CPU matrix.\n";
      std::cout << "Caught " << e.what() << "\n";
      diff_mats = 0;
    }
  }

  /************ RUNNING KERNELS *************/
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, 0); // We use 0 here because it is the "default" stream

  // gpu_rad_sweep6 should only be used when there are zeros in the middle of matrices,
  // otherwise gpu_rad_sweep5 is better. See gpu_funcs.cu
  if (8*(num_iters+2) < m && 8*(num_iters+2) < 49152/sizeof(double)) {
    gpu_rad_sweep6<<<n_blocks, threads, sizeof(double)*8*(num_iters+2)>>>(A_d, n, m, num_iters);
  } else {
    gpu_rad_sweep5<<<n_blocks, threads>>>(A_d, n, m, num_iters);
  }

  cudaEventRecord(computeFloatGpuEnd, 0);
  cudaEventSynchronize(computeFloatGpuStart);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime, computeFloatGpuStart, computeFloatGpuEnd);
  computeFloatGpuTime=(float)(computeFloatGpuElapsedTime)*0.001;

  // Transfer to RAM
  cudaMemcpy(A, A_d, sizeof(double)*n*m, cudaMemcpyDeviceToHost);

  // Print GPU_mat to file?
  if (write_mat == 1) print_sparse_matrix_to_file("data/sp_GPU" + std::to_string(n) + "x" + std::to_string(m) + "_p" + std::to_string(num_iters), A, n, m, num_iters);


  /************ CALCULATE AVERAGES *************/
  double avgC[n], avgA[n], avgD[n];

  // CPU avgs of GPU function
  gettimeofday(&t3, NULL);
  get_averages(A, n, m, avgA);
  gettimeofday(&t4, NULL);

  // GPU avgs of GPU function
  cudaEvent_t computeFloatGpuStart1, computeFloatGpuEnd1;
  float computeFloatGpuElapsedTime1,computeFloatGpuTime1;
  cudaEventCreate(&computeFloatGpuStart1);
  cudaEventCreate(&computeFloatGpuEnd1);
  cudaEventRecord(computeFloatGpuStart1, 0); 

  gpu_get_averages<<<blocks, threads>>>(A_d, n, m, avg_d);

  cudaEventRecord(computeFloatGpuEnd1, 0);
  cudaEventSynchronize(computeFloatGpuStart1);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd1); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime1, computeFloatGpuStart1, computeFloatGpuEnd1);
  computeFloatGpuTime1=(float)(computeFloatGpuElapsedTime1)*0.001;

  cudaMemcpy(avgD, avg_d, sizeof(double)*n, cudaMemcpyDeviceToHost);


  /************ COMPARE RESULTS ****************/
  std::cout << "\n=====>Errors\n";
  std::cout << "   Of matrix elements: \n";
  // Diff matrices - this will be avoided if there was an error reading in the comparison file
  if (diff_mats == 1) diff_matrices(A, C, n, m);

  if (avg == 1) {
    std::cout << "   Of GPU_avg function: \n";
    if (diff_mats == 1) diff_matrices(avgD, avgA, n, 1);
    get_averages(C, n, m, avgC);
    std::cout << "   Of averages (CPU vs GPU): \n";
    if (diff_mats == 1) diff_matrices(avgA, avgC, n, 1);
  }

  /************ PRINT TIMINGS ******************/
  double cpu_rad_time, cpu_avgs_time;

  if (print_time == 1) {
    if (n == 15360 && m == 15360 && num_iters == 500) {
      // This value is stored in the file cpu_timing_p500
      cpu_rad_time = 245.574;
    } else if (cpu_calc == 1) {
      cpu_rad_time = (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0);
    } else { 
      return 0;
    }

    printf("\n=====>Radiator timings (block size %d):\n\tCPU_timing: \t%lf\n\tGPU_timing: \t%lf\n\tSpeedup: \t%lf\n\n", block_size, cpu_rad_time, computeFloatGpuTime, cpu_rad_time/computeFloatGpuTime);

    if (avg == 1) {
      cpu_avgs_time = (double)(t4.tv_sec-t3.tv_sec)+((double)(t4.tv_usec - t3.tv_usec)/1000000.0); 
      printf("\n=====>Row average timings (block size %d):\n\tCPU_timing: \t%lf\n\tGPU_timing: \t%lf\n\tSpeedup: \t%lf\n\n", block_size, cpu_avgs_time, computeFloatGpuTime1, cpu_avgs_time/computeFloatGpuTime1);
    }
  }
  /*
     cudaError_t err = cudaGetLastError();  // add
     if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
     cudaProfilerStop();
   */

  free(A); free(C);
  cudaFree(A_d);

  return 0;
}

