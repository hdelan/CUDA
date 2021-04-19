#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#define BLOCK_SIZE 8

// KERNELS

__global__ void gpu_rad_sweep1(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  float * tmp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < n) {
    int index = idx*m;
    for (auto i=0;i<iters;++i) {
      // Begin sweep
      for (auto k=2;k<m-2;k++) {
        b_d[idx*m+k] = (1.70f*a_d[index+k-2] + 1.40f*a_d[index+k-1] + a_d[index+k] + 0.60f*a_d[index+k+1] + 0.30f*a_d[index+k+2])/5.0f;
      }
      b_d[index+m-2] = (1.70f*a_d[index+m-4] + 1.40f*a_d[index+m-3] + a_d[index+m-2] + 0.60f*a_d[index+m-1]+0.30f*a_d[index])/5.0f;
      b_d[index+m-1] = (1.70f*a_d[index+m-3] + 1.40f*a_d[index+m-2] + a_d[index+m-1] + 0.60f*a_d[index]+0.30f*a_d[index+1])/5.0f;
      // End sweep
      tmp = a_d;
      a_d = b_d;
      b_d = tmp;
    }
  }
}

__global__ void gpu_rad_sweep2(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  float * tmp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  // These values are reused so we will keep them in register
  float active_vals[5];
  if (idx < n) {
    int index = idx*m;
    for (auto i=0;i<iters;++i) {
      // Loading first values into registers
      for (auto i=0;i<5;i++) 
        active_vals[i] = a_d[index+i];

      // Start sweep
      for (auto k=2;k<m-3;k++) {
        b_d[index+k] = (1.70f*active_vals[(k-2)%5] + 1.40f*active_vals[(k-1)%5] + active_vals[k%5] + 0.60f*active_vals[(k+1)%5] + 0.30f*active_vals[(k+2)%5])/5.0f;
        active_vals[(k+3)%5] = a_d[index+k+3];
      }
      b_d[index+m-3] = (1.70f*active_vals[(m-5)%5] + 1.40f*active_vals[(m-4)%5] + active_vals[(m-3)%5] + 0.60f*active_vals[(m-2)%5] + 0.30f*active_vals[(m-1)%5])/5.0f;
      b_d[index+m-2] = (1.70f*active_vals[(m-4)%5] + 1.40f*active_vals[(m-3)%5] + active_vals[(m-2)%5] + 0.60f*active_vals[(m-1)%5] + 0.30f*((idx+1)/(float) n))/5.0f;
      b_d[index+m-1] = (1.70f*active_vals[(m-3)%5] + 1.40f*active_vals[(m-2)%5] + active_vals[(m-1)%5] + 0.60f*((idx+1)/(float)n) + 0.240f*(idx+1)/(float)n)/5.0f;
      // End sweep
      
      tmp = a_d;
      a_d = b_d;
      b_d = tmp;
    }
  }
}

__global__ void gpu_rad_sweep3(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  float * tmp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  // Using shared memory instead of registers.
  // The array is dynamically allocated with dimension blockDim.x*5
  extern __shared__ float active_vals_block[];
  
  // Easy index into thread's part of shared array
  float * active_vals = &active_vals_block[5*threadIdx.x];
  
  if (idx < n) {
    int index = idx*m;
    for (auto i=0;i<iters;++i) {
      // Loading first values into registers
      for (auto i=0;i<5;i++) 
        active_vals[i] = a_d[index+i];

      // Start sweep
      for (auto k=2;k<m-3;k++) {
        b_d[index+k] = (1.70f*active_vals[(k-2)%5] + 1.40f*active_vals[(k-1)%5] + active_vals[k%5] + 0.60f*active_vals[(k+1)%5] + 0.30f*active_vals[(k+2)%5])/5.0f;
        active_vals[(k+3)%5] = a_d[index+k+3];
      }
      b_d[index+m-3] = (1.70f*active_vals[(m-5)%5] + 1.40f*active_vals[(m-4)%5] + active_vals[(m-3)%5] + 0.60f*active_vals[(m-2)%5] + 0.30f*active_vals[(m-1)%5])/5.0f;
      b_d[index+m-2] = (1.70f*active_vals[(m-4)%5] + 1.40f*active_vals[(m-3)%5] + active_vals[(m-2)%5] + 0.60f*active_vals[(m-1)%5] + 0.30f*((idx+1)/(float) n))/5.0f;
      b_d[index+m-1] = (1.70f*active_vals[(m-3)%5] + 1.40f*active_vals[(m-2)%5] + active_vals[(m-1)%5] + 0.60f*((idx+1)/(float)n) + 0.240f*(idx+1)/(float)n)/5.0f;
      // End sweep
      
      tmp = a_d;
      a_d = b_d;
      b_d = tmp;
    }
  }
}


__global__ void gpu_rad_sweep4(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  __shared__ float pass_right[BLOCK_SIZE*2];
  __shared__ float pass_left[BLOCK_SIZE*2];
  constexpr int dim_per_thread = 15360/BLOCK_SIZE;
  float a_local[1920];
  float b_local[1920];

  // I need these pointers in order to 'swap' a_local and b_local
  float* tmp, *a_p=a_local, *b_p=b_local;

  int tidx = threadIdx.x;

  for (int i=0;i<dim_per_thread;i++)
    b_local[i] = 0.0f;

  pass_right[tidx*2] = 0.0f;
  pass_right[tidx*2+1] = 0.0f;

  pass_left[tidx*2] = 0.0f;
  pass_left[tidx*2+1] = 0.0f;

  __syncthreads();

  if (tidx == 0) {
    a_local[0] = b_local[0] = pass_left[BLOCK_SIZE-2] = (blockIdx.x + 1.0f) / (float) n;
    a_local[1] = b_local[1] = pass_left[BLOCK_SIZE-1] = 0.80f*(blockIdx.x + 1.0f) / (float) n;
  }
  __syncthreads();

  for (int j=0;j<iters;j++) {
    if (tidx != 0) {
      a_p[0] = (1.70f*pass_right[tidx*2] + 1.40f*pass_right[tidx*2+1] + b_p[0] + 0.60f*b_p[1] + 0.30f*b_p[2])/5.0f;
      a_p[1] = (1.70f*pass_right[tidx*2+1] + 1.40f*b_p[0] + b_p[1] + 0.60f*b_p[2] + 0.30f*b_p[3])/5.0f;
    }
    
    for (int i=2;i<dim_per_thread-2;i++)
      a_p[i] = (1.70f*b_p[i-2] + 1.40f*b_p[i-1] + b_p[i] + 0.60f*b_p[i+1] + 0.30f*b_p[i+2])/5.0f;
    
    a_p[dim_per_thread-2] = (1.70f*b_p[dim_per_thread-4] + 1.40f*b_p[dim_per_thread-3] + b_p[dim_per_thread-2] + 0.60f*b_p[dim_per_thread-1] + 0.30f*pass_left[tidx*2])/5.0f;
    a_p[dim_per_thread-1] = (1.70f*b_p[dim_per_thread-3] + 1.40f*b_p[dim_per_thread-2] + b_p[dim_per_thread-1] + 0.60f*pass_left[tidx*2] + 0.30f*pass_left[tidx*2+1])/5.0f;
    if (tidx != 0) {
      pass_left[(tidx-1)*2] = a_p[0];
      pass_left[(tidx-1)*2+1] = a_p[1];
    }

    if (tidx != BLOCK_SIZE-1) {
      pass_right[(tidx+1)*2] = a_p[dim_per_thread-2];
      pass_right[(tidx+1)*2+1] = a_p[dim_per_thread-1];
    }
    tmp = a_p;
    a_p = b_p;
    b_p = tmp;
    //__syncthreads();
  }
  // Writing back to global memory
  for (int i=0;i<dim_per_thread;i++)
    b_d[blockIdx.x*m + tidx*dim_per_thread + i] = b_local[i];
}
/*
__global__ void gpu_rad_sweep5(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  __shared__ float a_d[];
  
  for (int)
  
*/
