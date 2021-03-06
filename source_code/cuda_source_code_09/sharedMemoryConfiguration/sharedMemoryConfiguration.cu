//=============================================================================================
// Name        		: sharedMemory.cu
// Author      		: Jose Refojo
// Version     		:	29-06-2012
// Creation date	:	21-06-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use the shared memory
//=============================================================================================

#define BLOCK_SIZE 8
#include "stdio.h"

__device__ int returnBlockId () {
	return blockIdx.x;
}

// Shared memory is shared by all the threads in a block, and it is initialized (and used) inside a global
// or device function - we will use a global function here but the usage in the device functions is exactly
// the same
__global__ void scanTheadInformationGPU(int *threadIdsGPU,int *threadOtherIdsGPU,int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;


        __shared__ float sharedMemoryThread[BLOCK_SIZE];
	sharedMemoryThread[threadIdx.x] = threadIdx.x;

        __syncthreads();

	if ( idx <Ntot ) {
		threadIdsGPU[idx]=sharedMemoryThread[threadIdx.x];
		threadOtherIdsGPU[idx]=sharedMemoryThread[threadIdx.x]+blockIdx.x*blockDim.x;
		blockIdsGPU[idx]=returnBlockId();
	}
}


int main() {
	// pointers to host memory
	int *threadIds,*threadOtherIds, *blockIds;
	// pointers to device memory
	int *threadIdsGPU,*threadOtherIdsGPU, *blockIdsGPU;
	// N is the total size that we want
	int N=18;
	int i;

	// request 32KB shared memory / 32KB L1 cache configuration for all kernels
	// take note that cudaFuncCachePreferEqual should only work on cc 3.x
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	// configure scanTheadInformationGPU in particular to request more shared memory - 48KB shared memory / 16KB L1 cache
	cudaFuncSetCacheConfig(&scanTheadInformationGPU,cudaFuncCachePreferShared);

	// Allocate arrays threadIds and blockIds on host
	threadIds = (int*) malloc(N*sizeof(int));
	threadOtherIds = (int*) malloc(N*sizeof(int));
	blockIds = (int*) malloc(N*sizeof(int));

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIdsGPU, sizeof(int)*N);
	cudaMalloc ((void **) &threadOtherIdsGPU, sizeof(int)*N);
	cudaMalloc ((void **) &blockIdsGPU, sizeof(int)*N);

/*
	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Scan information from the threads
	scanTheadInformationGPU<<<dimGrid,dimBlock>>>(threadIdsGPU,threadOtherIdsGPU, blockIdsGPU, N);

	// Copy data from device memory to host memory
	cudaMemcpy(threadIds, threadIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadOtherIds, threadOtherIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockIds, blockIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	printf(" dimGrid=%d\n",dimGrid.x);
	for (i=0; i<N; i++) {
	       printf(" threadIds[%d]=%d\n",i,threadIds[i]);
	}
	for (i=0; i<N; i++) {
	       printf(" threadOtherIds[%d]=%d\n",i,threadOtherIds[i]);
	}
	for (i=0; i<N; i++) {
	       printf(" blockIds[%d]=%d\n",i,blockIds[i]);
	}

	// Free the memory
	free(threadIds);
	free(threadOtherIds);
	free(blockIds); 

	cudaFree(threadIdsGPU);
	cudaFree(threadOtherIdsGPU);
	cudaFree(blockIdsGPU);
}
