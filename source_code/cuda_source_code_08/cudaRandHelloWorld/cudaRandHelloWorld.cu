# include <stdio.h>
# include <stdlib.h>
# include <cuda.h>

// This is the include for the host curand
# include <curand.h>

// This is the include for the device curand
# include <curand_kernel.h>

__global__ void setup_kernel ( curandState * state ) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	// Each thread gets same seed, a different sequence number , no offset
	curand_init (1234 , idx , 0, &state[idx]);
}

__global__ void generate_kernel ( curandState *state , float* results ) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	// Copy state to local memory for efficiency - will be more useful 
	curandState localState = state [idx];

	// Generate pseudo - random from an uniform distribution
	results[idx] = (float)(curand_uniform (& localState ));

	// Copy state back to global memory
	state [idx] = localState ;
}
__global__ void generate_kernels ( curandState *state , float* results, int iterations ) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (iterations<1) return;

	// Copy state to local memory for efficiency - will be more useful 
	curandState localState = state [idx];

	// Generate pseudo - random unsigned ints
	results[idx]=0.0;
	for (int i=0;i<iterations;i++) {
		results[idx] += (float)(curand_uniform (& localState ));
	}
	results[idx]/=iterations;

	// Copy state back to global memory
	state [idx] = localState ;
}



__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		out[idx]=in1[idx]+in2[idx];
		//out[idx]=10;
	}
}

int main() {
	// pointers to host memory
	float *a, *b, *c;
	// pointers to device memory
	float *a_device, *b_device, *c_device;
	int N=10;
	int i;

	// Allocate arrays a, b and c on host
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	c = (float*) malloc(N*sizeof(float));

	// Allocate arrays a_device, b_device and c_device on device
	cudaMalloc ((void **) &a_device, sizeof(float)*N);
	cudaMalloc ((void **) &b_device, sizeof(float)*N);
	cudaMalloc ((void **) &c_device, sizeof(float)*N);

	// Initialize arrays a and b
	for (i=0; i<N; i++) {
		a[i]= (float) 2*i;
		b[i]=-(float) i;
	}

	// Copy data from host memory to device memory
	//cudaMemcpy(a_device, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	//cudaMemcpy(b_device, b, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Declare and initialize the pseudo-random number generator
	curandGenerator_t generator;
	curandCreateGenerator (&generator,CURAND_RNG_PSEUDO_DEFAULT);
	// Set the seed
	curandSetPseudoRandomGeneratorSeed (generator,1234ULL);
	// Generate N random numbers from an uniform distribution
	curandGenerateUniform (generator, c_device , N);

	// Copy data from device memory to host memory
	cudaMemcpy(c, c_device, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print c
	printf("cudaRandHelloWorld vector of random numbers generated from host:\n");
	for (i=0; i<N; i++) {
		printf("c[%2d](%10f)\n",i,c[i]);
	}

	// Device random stuff
	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	// Allocate deviceStates
	curandState *deviceStates;
	cudaMalloc (( void **) &deviceStates , N * sizeof ( curandState ));

	// Kernels to setup and generate the pseudo-random numbers on each thread
	setup_kernel <<<dimGrid,dimBlock>>> ( deviceStates );
	generate_kernel <<<dimGrid,dimBlock>>> ( deviceStates , a_device );

	// Copy data from device memory to host memory
	cudaMemcpy(a, a_device, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print a
	printf("cudaRandHelloWorld vector of random numbers generated in the device:\n");
	for (i=0; i<N; i++) {
		printf(" a[%2d](%10f)\n",i,a[i]);
	}

	generate_kernels <<<dimGrid,dimBlock>>> ( deviceStates , b_device, 10000 );

	// Copy data from device memory to host memory
	cudaMemcpy(b, b_device, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print b
	printf("cudaRandHelloWorld vector of random numbers generated in the device, should converge to 0.5:\n");
	for (i=0; i<N; i++) {
		printf(" b[%2d](%10f)\n",i,b[i]);
	}

	// Free the memory
	free(a); free(b); free(c);

	cudaFree (a_device);
	cudaFree (b_device);
	cudaFree (c_device);
	cudaFree (deviceStates);
}
