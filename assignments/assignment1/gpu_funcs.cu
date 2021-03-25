#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#include "matrix.h"

// KERNELS
__global__ void sum_abs_rows_GPU(float * data, float * rowsum, const int N, const int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < N) {
        	rowsum[idx] = 0.0f;
                for (int j = 0; j < M; ++j) {
                        rowsum[idx] += std::abs(data[idx*M + j]);
                }
        }
}

__global__ void sum_abs_cols_GPU(float * data, float * colsum, const int N, const int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < M) {
        	colsum[idx] = 0.0f;
                for (int j = 0; j < N; ++j) {
                        colsum[idx] += std::abs(data[j*M + idx]);
                }
        }
}

__global__ void reduce0_GPU(float * vector_GPU, const int N) {
	int stride = blockDim.x*gridDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int index = idx + stride;
	while (index < N) {
		vector_GPU[idx] += vector_GPU[index];
		index += stride;
	}
}


__global__ void reduce1_GPU(float * vector_GPU, const int N) {
	int stride = blockDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= stride || idx >= N) {
		return;
	}
	int index = idx + stride;
	while (index < blockDim.x*gridDim.x && index < N) {
		vector_GPU[idx] += vector_GPU[index];
		index += stride;
	}
	__syncthreads();
	if (idx == 0) {
		for (int i = 1; i < blockDim.x && i < N; i++)
			vector_GPU[idx] += vector_GPU[i];
	}
}

float vector_reduction_GPU(float * vector_GPU, const int N, dim3 dimBlock, dim3 dimGrid) {
	reduce0_GPU<<<dimGrid, dimBlock>>>(vector_GPU, N);
	reduce1_GPU<<<dimGrid, dimBlock>>>(vector_GPU, N);
	float ans;
	cudaMemcpy(&ans, vector_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	return ans;
}

