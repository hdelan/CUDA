// Redone simpleCUBLAS.cpp, comparing a matrix multiplication on the cpu, gsl-blas and cublas

// Includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Includes, cuda
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

// Includes, gsl blas
#include <gsl/gsl_cblas.h>

// Matrix size , hardcoded in this case
#define N  (4000)
#define displayN  (5)

// Host implementation of a simple version of sgemm
static void simple_sgemm(int n, float alpha, const float *A, const float *B,float beta, float *C) {
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;
            for (k = 0; k < n; ++k) {
                prod += A[j * n + k] * B[k * n + i];
            }
            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

// Main
int main(int argc, char **argv) {
	// Start the time tracking
	cudaEvent_t hostStart, hostFinish;
	cudaEvent_t blasStart, blasFinish;
	cudaEvent_t cuBlasStart, cuBlasFinish;

	cudaEventCreate(&hostStart);
	cudaEventCreate(&hostFinish);
	cudaEventCreate(&blasStart);
	cudaEventCreate(&blasFinish);
	cudaEventCreate(&cuBlasStart);
	cudaEventCreate(&cuBlasFinish);

	float hostElapsedTime=0.0;
	float blasElapsedTime=0.0;
	float cuBlasElapsedTime=0.0;

	// Arrays on the host memory
	float *A_cpu;
	float *B_cpu;
	float *C_cpu;
	float *C_ref_cpu;
	float *C_blas_cpu;

	// Arrays on device memory
	float *A_gpu = NULL;
	float *B_gpu = NULL;
	float *C_gpu = NULL;

	float alpha = 1.0f;
	float beta = 0.0f;
	int n2 = N * N;
	int i;
	unsigned int ui,uj;
	float error_norm;
	float ref_norm;
	float diff;

	// cuBLAS status and handler
	cublasStatus_t status;
	cublasHandle_t handle;

	// First, find the device
	int dev = findCudaDevice(argc, (const char**) argv);
	if( dev == -1 ) {
		return EXIT_FAILURE;
	}

	// Initialize CUBLAS
	printf("simpleCUBLAS test running..\n");
	status = cublasCreate(&handle);
	// Make sure that it was initialized properly
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	//Allocate host memory for the matrices, as usual
	A_cpu = (float *)malloc(n2 * sizeof(float));
	if (A_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (A)\n");
		return EXIT_FAILURE;
	}

	B_cpu = (float *)malloc(n2 * sizeof(float));
	if (B_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (B)\n");
		return EXIT_FAILURE;
	}

	C_cpu = (float *)malloc(n2 * sizeof(float));
	if (C_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}
	C_ref_cpu = (float *)malloc(n2 * sizeof(float));
	if (C_ref_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}
	C_blas_cpu = (float *)malloc(n2 * sizeof(float));
	if (C_blas_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	// Fill the matrices with test data
	for (ui = 0; ui < N; ui++) {
		for (uj = 0; uj < N; uj++) {
			A_cpu[uj*N+ui] = rand() / (float)RAND_MAX;
			B_cpu[uj*N+ui] = rand() / (float)RAND_MAX;
			//C_cpu[ui] = rand() / (float)RAND_MAX;
			//A_cpu[uj*N+ui] = (float)(ui)/(float)(N);
			//B_cpu[uj*N+ui] = 1.0;
			C_cpu[uj*N+ui] = 0.0;
			C_ref_cpu[uj*N+ui] = 0.0;
			C_blas_cpu[uj*N+ui] = 0.0;
		}
	}

	// Allocate device memory for the matrices, as usual, again
	if (cudaMalloc((void **)&A_gpu, n2 * sizeof(A_gpu[0])) != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&B_gpu, n2 * sizeof(B_gpu[0])) != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&C_gpu, n2 * sizeof(C_gpu[0])) != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
		return EXIT_FAILURE;
	}

	// Initialize the device matrices with the host matrices - we use cuBlas to do this
	status = cublasSetVector(n2, sizeof(A_cpu[0]), A_cpu, 1, A_gpu, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write A)\n");
		return EXIT_FAILURE;
	}

	status = cublasSetVector(n2, sizeof(B_cpu[0]), B_cpu, 1, B_gpu, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write B)\n");
		return EXIT_FAILURE;
	}

	status = cublasSetVector(n2, sizeof(C_cpu[0]), C_cpu, 1, C_gpu, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write C)\n");
		return EXIT_FAILURE;
	}

	// Performs operation using plain C code
	cudaEventRecord(hostStart, 0); // We use 0 here because it is the "default" stream
	simple_sgemm(N, alpha, A_cpu, B_cpu, beta,C_ref_cpu);
	cudaEventRecord(hostFinish, 0);
	cudaEventSynchronize(hostStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(hostFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&hostElapsedTime, hostStart, hostFinish);

	// Performs operation using gsl cblas
	cudaEventRecord(blasStart, 0); // We use 0 here because it is the "default" stream
	cblas_sgemm (	CblasRowMajor,
			CblasNoTrans,CblasNoTrans,
			N,N,N,1.0,
			A_cpu,N,B_cpu,N,0.0,C_blas_cpu,N);
	cudaEventRecord(blasFinish, 0);
	cudaEventSynchronize(blasStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(blasFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&blasElapsedTime, blasStart, blasFinish);

	// Performs operation using cublas
	cudaEventRecord(cuBlasStart, 0); // We use 0 here because it is the "default" stream
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B_gpu, N, A_gpu, N, &beta, C_gpu, N);
	cudaEventRecord(cuBlasFinish, 0); // We use 0 here because it is the "default" stream
	cudaEventSynchronize(cuBlasStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(cuBlasFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&cuBlasElapsedTime, cuBlasStart, cuBlasFinish);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! kernel execution error.\n");
		return EXIT_FAILURE;
	}

	// Allocate host memory for reading back the result from device memory, as usual
	C_cpu = (float *)malloc(n2 * sizeof(C_cpu[0]));

	if (C_cpu == 0) {
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	// Read the result back - again, we can use cublasGetVector to transfer the data
	status = cublasGetVector(n2, sizeof(C_cpu[0]), C_gpu, 1, C_cpu, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (read C)\n");
		return EXIT_FAILURE;
	}

	// Check result against reference
	error_norm = 0;
	ref_norm = 0;

	for (i = 0; i < n2; ++i) {
		diff = C_ref_cpu[i] - C_cpu[i];
		error_norm += diff * diff;
		ref_norm += C_ref_cpu[i] * C_ref_cpu[i];
	}

	error_norm = (float)sqrt((double)error_norm);
	ref_norm = (float)sqrt((double)ref_norm);

	if (fabs(ref_norm) < 1e-7) {
		fprintf(stderr, "!!!! reference norm is 0\n");
		return EXIT_FAILURE;
	}



	printf("A_cpu \n");
	for (ui=0;ui<displayN;ui++) {
		for (uj=0;uj<displayN;uj++) {
			printf("%10.5f",A_cpu[ui*N+uj]);
			if (uj<displayN-1) printf(",");
		}
		printf("\n");
	}
	printf("B_cpu \n");
	for (ui=0;ui<displayN;ui++) {
		for (uj=0;uj<displayN;uj++) {
			printf("%10.5f",B_cpu[ui*N+uj]);
			if (uj<displayN-1) printf(",");
		}
		printf("\n");
	}
	printf("C_ref_cpu \n");
	for (ui=0;ui<displayN;ui++) {
		for (uj=0;uj<displayN;uj++) {
			printf("%10.5f",C_ref_cpu[ui*N+uj]);
			if (uj<displayN-1) printf(",");
		}
		printf("\n");
	}
	printf("C_blas_cpu \n");
	for (ui=0;ui<displayN;ui++) {
		for (uj=0;uj<displayN;uj++) {
			printf("%10.5f",C_blas_cpu[ui*N+uj]);
			if (uj<displayN-1) printf(",");
		}
		printf("\n");
	}
	printf("C_cuBLAS \n");
	for (ui=0;ui<displayN;ui++) {
		for (uj=0;uj<displayN;uj++) {
			printf("%10.5f",C_cpu[ui*N+uj]);
			if (uj<displayN-1) printf(",");
		}
		printf("\n");
	}

	printf("Total time with host: %f, with gslblas: %f, with cuBlas: %f, speedup to host:%f, speedup to blas:%f\n",
		hostElapsedTime,blasElapsedTime,cuBlasElapsedTime,
		(float)(hostElapsedTime/cuBlasElapsedTime),(float)(blasElapsedTime/cuBlasElapsedTime));


	// Memory clean up - as usual for malloc'ed data
	free(A_cpu);
	free(B_cpu);
	free(C_cpu);
	free(C_ref_cpu);

	if (cudaFree(A_gpu) != cudaSuccess) {
		fprintf(stderr, "!!!! memory free error (A)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(B_gpu) != cudaSuccess) {
		fprintf(stderr, "!!!! memory free error (B)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(C_gpu) != cudaSuccess) {
		fprintf(stderr, "!!!! memory free error (C)\n");
		return EXIT_FAILURE;
	}

	// Shutdown
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	exit(error_norm / ref_norm < 1e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
