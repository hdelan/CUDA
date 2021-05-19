/**
 * \file:        gpu_funcs.cu
 * \brief:       Some kernels for cylindrical radiator finite differences
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-04-29
 */
#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#define SHARED_DIM 12288

// KERNELS

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function that exploits the sparseness of finite difference matrices
                 to perform a very fast cylindrical finite differences. 
                 
                 Uses two separate sets of computations/arrays. One for left propagation, and one
                 for right propagation.

                 Will only be called in main if we have space in shared memory for four separate arrays.

                 Thread safe.
 *
 * \param:       a_d
 * \param:       n
 * \param:       m
 * \param:       iters
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void gpu_rad_sweep6(float * a_d, unsigned int n, unsigned int m, unsigned int iters) {
  
  // Each array will have this dimension
  int shared_dims = 2*(iters+2);

  extern __shared__ float a_s[];
  
  // a_start, b_start will be for right propogation and 
  // a_end b_end will be for left propogation
  float * a_start = a_s;
  float * b_start = &a_s[2*(iters+2)];
  float * a_end = &a_s[4*(iters+2)];
  float * b_end = &a_s[6*(iters+2)];

  float * tmp;

  int tidx = threadIdx.x;
  int step = blockDim.x;
  int index = tidx;

  // Initializing the arrays
  while (index < shared_dims) {
    a_start[index] = b_start[index] = a_end[index] = b_end[index] = 0.0f;
    index += step;
  }

  a_start[0] = b_start[0] = a_end[shared_dims-2] = b_end[shared_dims-2] = (blockIdx.x+1)/ (float)n;
  a_start[1] = b_start[1] = a_end[shared_dims-1] = b_end[shared_dims-1] = 0.80f * a_start[0];


  for (int i=0;i<iters;i++) {
    __syncthreads();

    // Right propagation
    index = tidx + 2;
    while (index < shared_dims-2) {
      b_start[index] =  (1.70f*a_start[index-2] + 1.40f*a_start[index-1] + a_start[index] + 0.60f*a_start[index+1] + 0.30f*a_start[index+2])/5.0f;
      if (b_start[index]==0.0f) break;
      index += step;
    }
    
    // Left propagation
    index = shared_dims - 3 - tidx;
    while (index > 1) {
      b_end[index] =  (1.70f*a_end[index-2] + 1.40f*a_end[index-1] + a_end[index] + 0.60f*a_end[index+1] + 0.30f*a_end[index+2])/5.0f;
      if (b_end[index]==0.0f) break;
      index -= step;
    }
    
    // Swapping arrays
    tmp = a_start;
    a_start = b_start;
    b_start = tmp;

    tmp = a_end;
    a_end = b_end;
    b_end = tmp;
  }

  // Writing to global memory
  int glob_index = blockIdx.x*m + tidx;
  index = tidx;
  // Writing right prop
  while (index < shared_dims) {
    a_d[glob_index] = a_start[index];
    index += step;
    glob_index += step;
  }

  // Writing left prop
  glob_index = (blockIdx.x+1)*m - shared_dims + tidx+2;
  index = tidx;
  while (index < shared_dims - 2) {
    a_d[glob_index] = a_end[index];
    index += step;
    glob_index += step;
  }
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       Suitable for matrices that are non sparse. Loads entire rows into
                 shared memory by taking chunks of rows at a time. 
                
                 NB NOT THREAD SAFE for block size greater than 128

                 (due to lack of syncthreads in inner while loops)

                 Much slower than gpu_rad_sweep6 for lower iteration counts
 *
 * \param:       a_d
 * \param:       n
 * \param:       m
 * \param:       iters
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void gpu_rad_sweep5(float * a_d, unsigned int n, unsigned int m, unsigned int iters) {
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
    f1 = 0.80f*f0;

    glob_index = glob_start + tidx;

    //              BEGIN LOOP          //
    //  if entire row will not fit in SHARED_DIM     
    while (remaining > SHARED_DIM-2){
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
      //for (int i=0;i<loop_iters;i++) {
      while (shared_index < SHARED_DIM-2) {
        g0 = (1.70f*a_shared[shared_index-2] + 1.40f*a_shared[shared_index-1] + a_shared[shared_index] + 0.60f*a_shared[shared_index+1] + 0.30f*a_shared[shared_index+2])/5.0f;
        if (g1 >= 0.0f) a_shared[shared_index-step] = g1;
        // Swap g0 and g1 so that the just-computed value will be stored in the next cycle
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
    if (threadIdx.x == 0) {
      a_shared[0] = f0;
      a_shared[1] = f1;
    }
    __syncthreads();

    shared_index = tidx+2;
    glob_index = glob_start+tidx;

    // Load section of array into shared memory
    //while (glob_index < (blockIdx.x+1)*m) {
    while (shared_index < remaining+2){
      a_shared[shared_index] = a_d[glob_index];
      shared_index += step;
      glob_index += step;
    }

    // Setting endpoints to be a_d[0], a_d[1]
    a_shared[remaining+2] = (float)(blockIdx.x+1) / (float)n;
    a_shared[remaining+3] = 0.80f*(blockIdx.x+1) / (float)n;
    __syncthreads();

    shared_index = tidx+2;
    // Perform calculation from shared[2] to shared[SHARED_DIM-3]
    while (shared_index < remaining+2) {
      g0 = (1.70f*a_shared[shared_index-2] + 1.40f*a_shared[shared_index-1] + a_shared[shared_index] + 0.60f*a_shared[shared_index+1] + 0.30f*a_shared[shared_index+2])/5.0f;
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
    // Reset g0, g1 so not used on first iteration of next run
    g0 = -1.0f, g1 = -1.0f;

    shared_index = tidx+2;
    glob_index = glob_start + tidx;

    // Write shared array to global
    while (glob_index < (blockIdx.x+1)*m) {
      //while (shared_index < remaining+2){
      a_d[glob_index] = a_shared[shared_index];
      shared_index += step;
      glob_index += step;
    }
    __syncthreads();
    }
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       A simple row average function using the global memory.
                 
                 A faster function would exploit sparseness as in gpu_rad_sweep6 but
                 this runs quite fast anyway.
 *
 * \param:       A_d
 * \param:       n
 * \param:       m
 * \param:       avg_d
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void gpu_get_averages(float * A_d, unsigned int n, unsigned int m, float * avg_d) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float sum = 0.0f;
  if (idx < n) {
    for (int i=0;i<m;i++) sum += A_d[idx*m+i];
    avg_d[idx] = sum / (float) m;
  }
}
