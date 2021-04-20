#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <cuda_profiler_api.h> 

__global__ void gpu_rad_sweep1(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep2(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep3(float*, unsigned int, unsigned int, unsigned int, float*);
__global__ void gpu_rad_sweep4(float*, unsigned int, unsigned int, unsigned int, float*);

template <typename T>
void cpu_rad_sweep1(T*, unsigned int, unsigned int, unsigned int, T*);
template <typename T>
void cpu_rad_sweep2(T*, unsigned int, unsigned int, unsigned int, T*);
void get_averages(float * a, unsigned int n, unsigned int m, float * avg);

void diff_matrices(float *A, float *B, unsigned int n, unsigned int m);

void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, int & cpu_calc, unsigned int & block_size);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

int main(int argc, char * argv[]) { 
  unsigned int n {6400}, m {6400}, block_size {8}, max_iters {100};
  long unsigned int seed {123};
  int print_time {0}, cpu_calc {1};
  struct timeval t1, t2; //t3;
  parse_command_line(argc, argv, n, m, max_iters, seed, print_time, cpu_calc, block_size);

  std::cout << "n: " << n << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "max iters: " << max_iters << "\n";
  std::cout << "print_time: " << print_time << "\n";
  std::cout << "block_size: " << block_size << "\n";

  float * A = (float *) calloc(n*m, sizeof(float));
  float * B = (float *) calloc(n*m, sizeof(float));

  // Set boundary conditions
  for (auto i=0;i<n;++i) {
    A[i*n] = B[i*n] = 1.0f*(float)(i+1)/(float)n;
    A[i*n+1] = B[i*n+1] = 0.80f*(float)(i+1)/(float)n;
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

  std::cout << "Threads per block: " << threads.x << '\n';
  std::cout << "Num blocks: " << blocks.x << '\n';


  /*        CPU STUFF      */
  if (cpu_calc == 1) {
    //print_matrix_CPU(A, n, m);
    gettimeofday(&t1, NULL);
    cpu_rad_sweep1(A, n, m, max_iters, B);
    gettimeofday(&t2, NULL);
    //cpu_rad_sweep2(A, n, m, max_iters, B);
    //gettimeofday(&t3, NULL);
    printf("CPU Sweep 1 time %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    //printf("CPU Sweep 2 time %lf\n", (double)(t3.tv_sec-t2.tv_sec)+((double)(t3.tv_usec - t2.tv_usec)/1000000.0));
  } else {
    // Use gpu_rad_sweep1 as reference for accuracy?
    gpu_rad_sweep2<<<blocks, threads>>>(A_d, n, m, max_iters, B_d);
    // Move data to matrix B
    cudaMemcpy(B, A_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);
    // Reset device matrices
    cudaMemcpy(A_d, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
  }

  //cudaMemcpy(A, B_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

  //gpu_rad_sweep1<<<blocks, threads>>>(A_d, n, m, max_iters, B_d);
  //gpu_rad_sweep2<<<blocks, threads>>>(A_d, n, m, max_iters, B_d);
  //gpu_rad_sweep3<<<blocks, threads, 5*threads.x*sizeof(float)>>>(A_d, n, m, max_iters, B_d);
  dim3 n_blocks {n};
  gpu_rad_sweep4<<<n_blocks, threads>>>(A_d, n, m, max_iters, B_d);
  cudaMemcpy(A, B_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

  diff_matrices(A, B, n, m);

  float avg[n];
  get_averages(A, n, m, avg);
  std::cout << "\n\nCPU Matrix\n";
  print_matrix_CPU(A, n, m);
  std::cout << "\n\nGPU Matrix\n";
  print_matrix_CPU(B, n, m);
  std::cout << "Averages: \n";
  print_matrix_CPU(avg, n, 1);

  cudaError_t err = cudaGetLastError();  // add
  if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
  cudaProfilerStop();

  return 0;
}

void diff_matrices(float *A, float *B, unsigned int n, unsigned int m) {
  unsigned int index_r, index_c;
  float max_diff = 0.0f;
  for (int i=0;i<n*m;i++) {
    if (fabs(A[i] - B[i]) > max_diff) {
      max_diff = fabs(A[i] - B[i]);
      index_r = i / m;
      index_c = i % m;
    }
  }
  std::cout << "Max diff: " << max_diff << " at index (" << index_r << ", " << index_c << ")\n";
  std::cout << "A[i]: " << A[index_r*m+index_c] << "\nB[i]: " << B[index_r*m+index_c] << "\n";
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

