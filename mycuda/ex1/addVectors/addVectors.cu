#include <stdio.h>
#include <time.h>
__global__ void vecAdd(float * in1, float * in2, float * out, long int Ntotal) {
	long int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Ntotal){
		out[index] = in1[index] + in2[index];
	}
	return;
}

__global__ void vecTriad(float * in1, float * in2, float * in3, float * out, long int Ntotal) {
	long int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Ntotal) {
		out[index] = in1[index]*in2[index] + in3[index];
	}
	return;
}


int main(void) {
	
	long int N = 100000000;
	float* a =(float *) malloc(sizeof(float)*N);
	float* b =(float *) malloc(sizeof(float)*N);
	float* c =(float *) malloc(sizeof(float)*N);
	float* d =(float *) malloc(sizeof(float)*N);
	
	for (long int i = 0; i < N; ++i) {
		a[i] = i*2;
		b[i] = i*3;
		c[i] = i*5;
	}
	printf("Initialized\n");
	clock_t cudaStart = clock();

	float *a_d, *b_d, *c_d, *d_d;

	cudaMalloc((void **) &a_d, sizeof(float)*N);
	cudaMalloc((void **) &b_d, sizeof(float)*N);
	cudaMalloc((void **) &c_d, sizeof(float)*N);
	cudaMalloc((void **) &d_d, sizeof(float)*N);

	printf("CUDA malloced\n");
	cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c, sizeof(float)*N, cudaMemcpyHostToDevice);

	long int block_size=2048;
	dim3 dimBlock(block_size);
	dim3 dimGrid(N/dimBlock.x + (!(N%dimBlock.x)?0:1));

	vecTriad<<<dimGrid, dimBlock>>>(a_d, b_d, c_d,d_d, N);
	
	cudaMemcpy(d, d_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	
	
	printf("Cuda time taken = %lf\n", (double)(clock() - cudaStart)/CLOCKS_PER_SEC);

	for (int i = 0; i < 10; i ++) {
		printf("a[%d](%f) * b[%d](%f) + c[%d](%f) = d[%d](%f)\n", i, a[i], i, b[i],i,c[i], i, d[i]);
	}



	clock_t serialStart = clock();
	for (long int i = 0; i < N; i++){
		d[i] = a[i]*b[i] + c[i];
	}
	printf("Serial time taken = %lf\n", (double)(clock() - serialStart)/CLOCKS_PER_SEC);

	for (int i = 0; i < 10; i ++) {
		printf("a[%d](%f) * b[%d](%f) + c[%d](%f) = d[%d](%f)\n", i, a[i], i, b[i],i,c[i], i, d[i]);
	}
	


	free(a); free(b); free(c);
	cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
	return 0;
}
