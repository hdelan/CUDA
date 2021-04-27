#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <cuda_profiler_api.h> 

#include "blockdim.h"

__global__ void gpu_rad_sweep1(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep2(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep3(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep4(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep5(float*, unsigned int, unsigned int, unsigned int, float*);

void print_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M);
void read_matrix_from_file(std::string filename, float * A);

template <typename T>
void cpu_rad_sweep1(T*, unsigned int, unsigned int, unsigned int, T*);
template <typename T>
void cpu_rad_sweep2(T*, unsigned int, unsigned int, unsigned int, T*);
void get_averages(float * a, unsigned int n, unsigned int m, float * avg);

void diff_matrices(float *A, float *B, unsigned int n, unsigned int m);

void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, int & cpu_calc, unsigned int & block_size, int & write_file);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

int main(int argc, char * argv[]) { 
  unsigned int n {15360}, m {15360}, block_size {BLOCK_SIZE}, num_iters {500};
  long unsigned int seed {123};
  int print_time {0}, cpu_calc {1}, diff_mats {1}, write_mat {0};
  struct timeval t1, t2; //t3;
  parse_command_line(argc, argv, n, m, num_iters, seed, print_time, cpu_calc, block_size, write_mat);

  std::cout << "n: " << n << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "max iters: " << num_iters << "\n";
  std::cout << "print_time: " << print_time << "\n";
  std::cout << "block_size: " << block_size << "\n";
  // A 
  float * A = (float *) calloc(n*m, sizeof(float));
  float * C = (float *) calloc(n*m, sizeof(float));

  // Set boundary conditions
  for (auto i=0;i<n;++i) {
    A[i*m] = C[i*m] = 1.0f*(float)(i+1)/(float)n;
    A[i*m+1] = C[i*m+1] = 0.80f*(float)(i+1)/(float)n;
  }

  float *A_d, *B_d, *avg_d;
  cudaMalloc((void **) &A_d, sizeof(float)*n*m);
  cudaMalloc((void **) &B_d, sizeof(float)*n*m);
  cudaMalloc((void **) &avg_d, sizeof(float)*n);
  cudaMemcpy(A_d, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
  
  // Initialize block_size
  dim3 threads {block_size};
  dim3 blocks {(n/threads.x) + (!(n%threads.x)?0:1)};
  dim3 n_blocks {n};

  std::cout << "Threads per block: " << threads.x << '\n';
  std::cout << "Num blocks: " << blocks.x << '\n';


  /*        CPU STUFF      */
  // This file will be written to if the -c flag is not provided, or it will be read from
  std::string filename {"data/CPU" + std::to_string(n) + "x" + std::to_string(m) + "_p" + std::to_string(num_iters) + ".txt"};
  
  if (cpu_calc == 1) {
    gettimeofday(&t1, NULL);
    cpu_rad_sweep1(C, n, m, num_iters, A);
    gettimeofday(&t2, NULL);
    printf("CPU Sweep 1 time %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    print_matrix_to_file(filename, A, n, m);
  } else {
    try {
      std::cout << "Reading CPU-generated comparison matrix from file.\n";
      gettimeofday(&t1, NULL);
      read_matrix_from_file(filename, C);
      gettimeofday(&t2, NULL);
      printf("Finished reading matrix from file. Time taken: %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    } catch (std::exception e) {
      std::cout << "Could not read from file! Will not be able to compare GPU matrix with CPU matrix.\n";
      std::cout << "Caught " << e.what() << "\n";
      diff_mats = 0;
    }
  }

  //cudaMemcpy(A, B_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

  //gpu_rad_sweep1<<<blocks, threads>>>(A_d, n, m, num_iters, B_d);
  //gpu_rad_sweep2<<<blocks, threads>>>(A_d, n, m, num_iters, B_d);
  //gpu_rad_sweep3<<<blocks, threads, 5*threads.x*sizeof(float)>>>(A_d, n, m, num_iters, B_d);
  gpu_rad_sweep5<<<n_blocks, threads>>>(A_d, n, m, num_iters, B_d);

  cudaMemcpy(A, A_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

  // Print GPU_mat to file
  if (write_mat == 1) print_matrix_to_file(filename, A, n, m);

  std::cout << "=====>Error of individual terms: \n";
  // Diff matrices - this will be avoided if there was an error reading in the comparison file
  if (diff_mats == 1) diff_matrices(A, C, n, m);
  

  float avgC[n], avgA[n];
  get_averages(A, n, m, avgA);
  get_averages(C, n, m, avgC);
  std::cout << "=====>Error of averages: \n";
  if (diff_mats == 1) diff_matrices(avgA, avgC, n, 1);
  
  std::cout << "\n\nGPU Matrix\n";
  print_matrix_CPU(A, n, m);
  std::cout << "\n\nCPU Matrix\n";
  print_matrix_CPU(C, n, m);
  //std::cout << "Averages: \n";
  //print_matrix_CPU(avg, n, 1);

  cudaError_t err = cudaGetLastError();  // add
  if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
  cudaProfilerStop();

  free(A); free(C);

  return 0;
}

void diff_matrices(float *A, float *C, unsigned int n, unsigned int m) {
  unsigned int index_r=0, index_c=0, count=0, gpu_bigger=0;
  float max_diff = 0.0f, diff = 0.0f;
  for (int i=0;i<n*m;i++) {
    diff = fabs(A[i] - C[i]);
    if (diff > 0.00001f) {
      if (A[i] > C[i]) gpu_bigger++;
      count++;
    }
    if (diff > max_diff) {
      max_diff = diff;
      index_r = i / m;
      index_c = i % m;
    }
  }
  std::cout << "Difference in entries greater than 1e-5 at " << count << " of " << n*m << " points\n";
  std::cout << "GPU bigger at " << gpu_bigger << " of " << count << " points.\n";
  std::cout << "Max diff: " << max_diff << " at index (" << index_r << ", " << index_c << ")\n";
  std::cout << "GPU_mat[i]: " << A[index_r*m+index_c] << "\nCPU_mat[i]: " << C[index_r*m+index_c] << "\n";
}


/*
   std::cout << "Rowsum" << std::setw(12) << std::setprecision(10) 
   << rowsum_time_CPU << std::setw(12) 
   << rowsum_time_GPU << std::setw(15) 
   << rowsum_time_CPU/rowsum_time_GPU << std::setw(18) 
   << std::fabs((sum_of_rowsums_GPU-sum_of_rowsums_CPU)/sum_of_rowsums_CPU) << std::endl;
   std::cout << "Colsum"  << std::setw(12) 
   << colsum_time_CPU << std::setw(12) 
   << colsum_time_GPU << std::setw(15) 
   << colsum_time_CPU/colsum_time_GPU << std::setw(18) 
   << std::fabs((sum_of_colsums_GPU-sum_of_colsums_CPU)/sum_of_colsums_CPU) << std::endl;
   std::cout << "Reduce" <<  std::setw(12)
   << reduction_time_CPU <<std::setw(12) 
   << reduction_time_GPU << std::setw(15) 
   << reduction_time_CPU/reduction_time_GPU << std::setw(18) << std::endl;

   }

   std::cout << '\n';

   return 0;
   }
 */


