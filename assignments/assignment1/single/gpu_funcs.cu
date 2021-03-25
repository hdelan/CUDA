/**
 * \file:        gpu_funcs.cu
 * \brief:       Some GPU funcs for CUDA assignment 1
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-03-25
 */
#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#include "matrix.h"

// KERNELS

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A kernel to compute the rowsums on the GPU using only global memory
 *
 * \param:       matrix_GPU         The matrix to be rowsummed (in device memory)
 * \param:       rowsum_GPU         The N-length array to be returned (in device memory)
 * \param:       N
 * \param:       M
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void sum_abs_rows_GPU(float * matrix_GPU, float * rowsum_GPU, const int N, const int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < N) {
        	rowsum_GPU[idx] = 0.0f;
                for (int j = 0; j < M; ++j) {
                        rowsum_GPU[idx] += std::abs(matrix_GPU[idx*M + j]);
                }
        }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A kernel to compute the colsums on the GPU using only global memory
 *
 * \param:       matrix_GPU         The matrix to be colsummed (in device memory)
 * \param:       colsum_GPU         The M-length array to be returned (in device memory)
 * \param:       N
 * \param:       M
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void sum_abs_cols_GPU(float * matrix_GPU, float * colsum_GPU, const int N, const int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < M) {
        	colsum_GPU[idx] = 0.0f;
                for (int j = 0; j < N; ++j) {
                        colsum_GPU[idx] += std::abs(matrix_GPU[j*M + idx]);
                }
        }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       First part of the vector_reduction_GPU function. This gathers all of 
                 the vector into the first NUM_THREADS indices in the same vector
 *
 * \param:       vector_GPU
 * \param:       N
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void reduce0_GPU(float * vector_GPU, const int N) {
	int stride = blockDim.x*gridDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int index = idx + stride;
	while (index < N) {
		vector_GPU[idx] += vector_GPU[index];
		index += stride;
	}
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       The second part in the reduction function. This function is 
                 only executed by a single block and sums the values in the first 
                 NUM_THREADS indices of the array into the first blockDimx.x indices of the
                 array, before the thread in index 0 sums the remaining blockDimx.x values into
                 vector[0]
 *
 * \param:       vector_GPU
 * \param:       N
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void reduce1_GPU(float * vector_GPU, const int N) {
	int stride = blockDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

        // Only to be executed by first block!
	if (idx >= stride || idx >= N) {
		return;
	}

	int index = idx + stride;
	while (index < blockDim.x*gridDim.x && index < N) {
		vector_GPU[idx] += vector_GPU[index];
		index += stride;
	}

	__syncthreads();

        // Sum final blockDimx.x entries into vector_GPU[0]
	if (idx == 0) {
		for (int i = 1; i < blockDim.x && i < N; i++)
			vector_GPU[idx] += vector_GPU[i];
	}
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A wrapper function around reduce0_GPU and reduce1_GPU
 *
 * \param:       vector_GPU
 * \param:       N
 * \param:       dimBlock
 * \param:       dimGrid
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
float vector_reduction_GPU(float * vector_GPU, const int N, dim3 dimBlock, dim3 dimGrid) {
	reduce0_GPU<<<dimGrid, dimBlock>>>(vector_GPU, N);
	reduce1_GPU<<<dimGrid, dimBlock>>>(vector_GPU, N);
	float ans;
	cudaMemcpy(&ans, vector_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	return ans;
}

