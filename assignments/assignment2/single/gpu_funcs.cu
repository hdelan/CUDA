#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#define BLOCK_SIZE 32
#define SHARED_DIM 12288

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

__global__ void gpu_rad_sweep5(float * a_d, unsigned int n, unsigned int m, unsigned int iters, float * b_d) {
  __shared__ float a_shared[SHARED_DIM];
  int tidx = threadIdx.x;
  int step = blockDim.x;

  // These values will be used so we only need one array to do our calculations, instead
  // of two
  float f0, f1;
  float g0 = -1.0, g1 = -1.0, tmp;

  int remaining;
  int glob_index, shared_index, glob_start;

  for (unsigned int i=0;i<iters;i++) {

    glob_start = m*blockIdx.x+2;;
    remaining = m-2;
    f0 = (blockIdx.x+1)/ (float)n;
    f1 = 0.80f*(blockIdx.x+1)/ (float)n;
    
    glob_index = glob_start + tidx;

    //              BEGIN LOOP          //
    //  if entire row will not fit in SHARED_DIM     
    while (remaining > SHARED_DIM) {
      // These values will be cached from previous cycle or will hold boundary conditions
      a_shared[0] = f0;
      a_shared[1] = f1;

      shared_index = tidx+2;
      glob_index = glob_start+tidx;

      // Load section of array into shared memory
      while (shared_index < SHARED_DIM) {
        a_shared[shared_index] = a_d[glob_index];
        shared_index += step;
        glob_index += step;
      }

      __syncthreads();

      shared_index = tidx+2;

      // Perform calculation from shared[2] to shared[SHARED_DIM-3]
      while (shared_index < SHARED_DIM-2) {
        g0 = (1.70f*a_shared[shared_index-2] + 1.40f*a_shared[shared_index-1] + a_shared[shared_index] + 0.60f*a_shared[shared_index+1] + 0.30f*a_shared[shared_index+2])/5.0f;
        // TODO come up with a smart way of syncing threads
        //__syncthreads();
        if (g1 >= 0.0f) a_shared[shared_index-step] = g1;
        // Swap g0 and g1 so that the just computed value will be stored in the next cycle
        tmp = g0;
        g0 = g1;
        g1 = tmp;
        shared_index +=step;
      }
      __syncthreads();
      // Cache 4th and 3rd last values of prev array to store in first two vals of next array
      f0 = a_shared[SHARED_DIM-4];
      f1 = a_shared[SHARED_DIM-3];

      // Store final vals in shared array
      if (shared_index-step < SHARED_DIM) a_shared[shared_index-step] = g1;
      __syncthreads();
      
      // Reset g0, g1 so not used on first iteration of next run
      g1 = -1.0f, g0 = -1.0f;

      shared_index = tidx+2;
      glob_index = glob_start+tidx;
      // Write shared array to global
      while (shared_index < SHARED_DIM-2) {
        a_d[glob_index] = a_shared[shared_index];
        shared_index += step;
        glob_index += step;
      }
      __syncthreads();

      // Decrement the global index so last two values of prev array are reloaded for simplicity
      remaining -= SHARED_DIM - 4;
      glob_start += SHARED_DIM - 4;
    }
    //            END LOOP             //
    // The rest of array is now smaller than shared dim
    // These values will be cached from previous cycle or will hold boundary conditions
    a_shared[0] = f0;
    a_shared[1] = f1;
    __syncthreads();

    shared_index = tidx+2;

    // Load section of array into shared memory
    while (glob_index < (blockIdx.x+1)*m) {
      a_shared[shared_index] = a_d[glob_index];
      shared_index += step;
      glob_index += step;
    }

    // Setting endpoints to be a_d[0], a_d[1]
    a_shared[remaining+2]  = (blockIdx.x+1)/ (float)n;
    a_shared[remaining+3] = 0.80f*(blockIdx.x+1)/ (float)n;

    __syncthreads();

    shared_index = tidx+2;
    // Perform calculation from shared[2] to shared[SHARED_DIM-3]
    while (shared_index < remaining+2) {
      g0 = (1.70f*a_shared[shared_index-2] + 1.40f*a_shared[shared_index-1] + a_shared[shared_index] + 0.60f*a_shared[shared_index+1] + 0.30f*a_shared[shared_index+2])/5.0f;
      //__syncthreads();
      if (g1 >= 0.0f) a_shared[shared_index-step] = g1;
      // Swap g0 and g1 so that the just computed value will be stored in the next cycle
      tmp = g0;
      g0 = g1;
      g1 = tmp;
      shared_index +=step;
    }

    __syncthreads();
    // Store final vals in shared array
    if (g1 >= 0.0f) a_shared[shared_index-step] = g1;
    __syncthreads();
    // Reset g1 so not used on first iteration of next run
    g1 = -1.0f;

    shared_index = tidx+2;
    glob_index = glob_start + tidx;

    // Write shared array to global
    while (glob_index < (blockIdx.x+1)*m) {
      a_d[glob_index] = a_shared[shared_index];
      shared_index += step;
      glob_index += step;
    }
    __syncthreads();
  }
}


