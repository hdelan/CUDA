/**
 * \file:        gpu_funcs.cu
 
 * \brief:       Some gpu functions to calculate the exponential integral function.
                 
                 Uses GPU_exponentialIntegralFloat3 as a base. 
                 
                 Other functions are wrappers to run and time this kernel on one 
                 or two cards.

 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-05-24
 */

#include "gpu_funcs.h"
#include "cpu_funcs.h"

// These uses of constant memory were slower than just hardcoding values into
// functions
// extern __constant__ float C_eulerConstant; 
// extern __constant__ float epsilon;
// extern __constant__ float bigFloat;

using namespace std;

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Calculating the exponential integral function on a 2d grid
 *
 * \param:       start
 * \param:       end
 * \param:       num_samples
 * \param:       start_n
 * \param:       max_n
 * \param:       division
 * \param:       A
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void GPU_exponentialIntegralFloat_3 (const float start, const float end, const int num_samples, const int start_n, const int max_n, float division, float * A) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        int n=idy+start_n;
        float x=(idx)*division+start;
        __shared__ float eulerConstant; eulerConstant=0.5772156649015329f;
        __shared__ float psi;
        __shared__ float epsilon; epsilon=1.E-30;
        __shared__ float bigFloat; bigFloat=1.E100;
        int i,ii,nm1=n-1;
        float a=start,b=end,c,d,del,fact,h,ans=0.0f;
        int dev;
        cudaGetDevice(&dev);
        //if (blockIdx.x+threadIdx.x==0) printf("Running for device %d\n", dev);
        
        if (idx >= num_samples || n >= max_n) return;

        if (n==0) {
                A[idy*num_samples+idx] = exp(-x)/x;
                return;
        } else {
                if (x<=1.0) {
                        ans=(nm1!=0 ? 1.0f/nm1 : -log(x)-eulerConstant);	// First term
                        fact=1.0f;
                        for (i=1;i<=20000000;i++) {
                                fact*=-x/i;
                                if (i != nm1) {
                                        del = -fact/(i-nm1);
                                } else {
                                        psi = -eulerConstant;
                                        for (ii=1;ii<=nm1;ii++) {
                                                psi += 1.0f/ii;
                                        }
                                        del=fact*(-log(x)+psi);
                                }
                                ans+=del;
                                if (fabs(del)<fabs(ans)*epsilon) {
                                        A[idy*num_samples+idx] = ans;
                                        return;
                                }
                        }
                        //if (i==2000000) printf("Series failed in exponential integral");
                }
                else {
                        b=x+n;
                        c=bigFloat;
                        d=1.0f/b;
                        h=d;
                        for (i=1;i<=20000000;i++) {
                                a=-i*(nm1+i);
                                b+=2.0f;
                                d=1.0f/(a*d+b);
                                c=b+a/c;
                                del=c*d;
                                h*=del;
                                if (fabs(del-1.0f)<=epsilon) {
                                        ans=h*exp(-x);
                                        A[idy*num_samples+idx] = ans;
                                        return;
                                }
                        }
                }
        }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A wrapper function to launch kernel on two cards. We are going 
                 to split the 1d array in half which corresponds to a horizontal 
                 split, so one card will calculate the entire interval from n=1 to 
                 n=n/2 and another will calculate from n=n/2 to n=n
 *
 * \param:       resultsFloatGpu
 * \param:       n
 * \param:       numberOfSamples
 * \param:       a
 * \param:       b
 * \param:       division
 * \param:       block_size
 * \param:       yblock
 * \param:       time_taken
 */
/* ----------------------------------------------------------------------------*/
void launch_on_two_cards(float * & resultsFloatGpu, const unsigned n, const unsigned numberOfSamples, const float a, const float b, const float division, const unsigned block_size, const unsigned yblock, float & time_taken) {

  // Create streams
  cudaStream_t stream[2];


  int count;
  cudaGetDeviceCount(&count);
  printf("Number of graphics cards: %d\n", count);

  dim3 blocks {numberOfSamples/block_size+1, n/(2*yblock)+1}, threads {block_size, yblock};

  float *A_d0, *A_d1;

  /*************** CARD 1 **************/
  cudaSetDevice(0);
  cudaStreamCreate(&stream[0]);

  // TIMINGS
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, stream[0]); // We use 0 here because it is the "default" stream

  cudaMalloc((void **) &A_d0, sizeof(float)*numberOfSamples*(n/2));
  GPU_exponentialIntegralFloat_3<<<blocks, threads, 0, stream[0]>>>(a+division, b, numberOfSamples,1,n/2+1, division,A_d0);
  cudaMemcpyAsync(resultsFloatGpu, A_d0, sizeof(float)*numberOfSamples*n/2, cudaMemcpyDeviceToHost, stream[0]);

  cudaEventRecord(computeFloatGpuEnd, stream[0]);
  cudaEventSynchronize(computeFloatGpuStart);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime, computeFloatGpuStart, computeFloatGpuEnd);
  computeFloatGpuTime=(float)(computeFloatGpuElapsedTime)*0.001;
  if (timing) printf("GPU timing for card 1: %E seconds\n", computeFloatGpuTime);

  /*************** CARD 2 **************/
  cudaSetDevice(1);
  cudaStreamCreate(&stream[1]);

  // TIMINGS
  cudaEvent_t computeFloatGpuStart1, computeFloatGpuEnd1;
  float computeFloatGpuElapsedTime1,computeFloatGpuTime1;
  cudaEventCreate(&computeFloatGpuStart1);
  cudaEventCreate(&computeFloatGpuEnd1);
  cudaEventRecord(computeFloatGpuStart1, stream[1]); // We use 0 here because it is the "default" stream

  cudaMalloc((void **) &A_d1, sizeof(float)*numberOfSamples*(n/2 + (!(n%2)?0:1)));
  GPU_exponentialIntegralFloat_3<<<blocks, threads,0,stream[1]>>>(a+division, b, numberOfSamples,n/2+1,n+1,division,A_d1);
  cudaMemcpyAsync(resultsFloatGpu+(numberOfSamples)*n/2, A_d1, sizeof(float)*numberOfSamples*(n/2 + (!(n%2)?0:1)), cudaMemcpyDeviceToHost, stream[1]);

  cudaEventRecord(computeFloatGpuEnd1, stream[1]);
  cudaEventSynchronize(computeFloatGpuStart1);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd1); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime1, computeFloatGpuStart1, computeFloatGpuEnd1);
  computeFloatGpuTime1=(float)(computeFloatGpuElapsedTime1)*0.001;
  
  if (timing) printf("GPU timing for card 2: %E seconds\n", computeFloatGpuTime1);
  time_taken = max(computeFloatGpuTime, computeFloatGpuTime1);
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       A wrapper function to launch kernel on one card
 *
 * \param:       resultsFloatGpu       output
 * \param:       n
 * \param:       numberOfSamples
 * \param:       a
 * \param:       b
 * \param:       division
 * \param:       block_size
 * \param:       yblock
 * \param:       time_taken             output
 */
/* ----------------------------------------------------------------------------*/
void launch_on_one_card(float * & resultsFloatGpu, const unsigned n, const unsigned numberOfSamples, const float a, const float b, const float division, const unsigned block_size, const unsigned yblock, float & time_taken) {

  bool dynamic {false};

  // TIMINGS
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, 0); // We use 0 here because it is the "default" stream
          
  float *A_d;
  cudaMalloc((void **) &A_d, sizeof(float)*numberOfSamples*n);
  
  // RUN KERNEL
  if (dynamic) {
          printf("Running dynamically parallel kernel on 1 card\n");
          dim3 blocks {n/block_size+1}, threads {block_size};
          GPU_exponentialIntegralFloat_4_launch<<<blocks,threads>>> 
                  (a+division, b, numberOfSamples, 1, n+1, division, A_d);
  } else {

          dim3 blocks {numberOfSamples/block_size+1, n/yblock+1}, threads {block_size, yblock};
          GPU_exponentialIntegralFloat_3<<<blocks, threads>>>
                  (a+division, b, numberOfSamples, 1, n+1, division, A_d);
  }
  cudaMemcpy(resultsFloatGpu, A_d, sizeof(float)*numberOfSamples*n, cudaMemcpyDeviceToHost);

  // TIMINGS
  cudaEventRecord(computeFloatGpuEnd, 0);
  cudaEventSynchronize(computeFloatGpuStart);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime, computeFloatGpuStart, computeFloatGpuEnd);
  computeFloatGpuTime=(float)(computeFloatGpuElapsedTime)*0.001f;

  if (timing) printf("Total GPU timings: %E seconds\n", computeFloatGpuTime);
  time_taken = computeFloatGpuTime;
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       Part 1 in dynamically parallel solution. Each thread in this 
                 kernel is responsible for an entire n value (column). It will 
                 precompute psi to pass into its child kernels. The child kernels
                 will compute the function for each value of x in our range.
                 
 *
 * \param:       start
 * \param:       end
 * \param:       num_samples
 * \param:       start_n
 * \param:       max_n
 * \param:       division
 * \param:       A
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void GPU_exponentialIntegralFloat_4_launch (const float start, const float end, const int num_samples, const int start_n, const int max_n, float division, float * A) {
        int n=blockIdx.x*blockDim.x+threadIdx.x+1;
        if (n >= max_n) return;
        float psi=-0.5772156649015329f;
        if (start <= 1.0f) {
                for (int ii=1;ii<=n-1;ii++) {
                        psi += 1.0f/ii;
                }
        } else {
                psi = 0.0;
        }
        GPU_exponentialIntegralFloat_4_execute<<<num_samples/blockDim.x+1, blockDim.x>>>
                (start, end, num_samples, n, division, psi, A);
         

}



/* --------------------------------------------------------------------------*/
/**
 * \brief:       Part 2 in dynamically parallel solution. Values of n are passed 
                 down and this kernel computes for various values of x.
 *
 * \param:       start
 * \param:       end
 * \param:       num_samples
 * \param:       n
 * \param:       division
 * \param:       psi_precomputed
 * \param:       A
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
__global__ void GPU_exponentialIntegralFloat_4_execute (const float start, const float end, const int num_samples, const int n, const float division, const float psi_precomputed, float * A) {
        __shared__ float eulerConstant; eulerConstant=0.5772156649015329f;
        __shared__ float psi; psi=psi_precomputed;
        __shared__ float epsilon; epsilon=1.E-30;
        __shared__ float bigFloat; bigFloat=1.E100;
        int i,nm1=n-1;
        float a=start,b=end,c,d,del,fact,h,ans=0.0f;

        int idx=blockIdx.x*blockDim.x+threadIdx.x;

        float x = idx*division+start;

        if (idx >= num_samples || n==0) return;

        if (x<=1.0f) {
                ans=(nm1!=0 ? 1.0f/nm1 : -log(x)-eulerConstant);	// First term
                fact=1.0f;
                for (i=1;i<=20000000;i++) {
                        fact*=-x/i;
                        if (i != nm1) {
                                del = -fact/(i-nm1);
                        } else {
                                del=fact*(-log(x)+psi);
                        }
                        ans+=del;
                        if (fabs(del)<fabs(ans)*epsilon) {
                                A[nm1*num_samples+idx] = ans;
                                return;
                        }
                }
                //if (i==2000000) printf("Series failed in exponential integral");
        }
        else {
                b=x+n;
                c=bigFloat;
                d=1.0f/b;
                h=d;
                for (i=1;i<=20000000;i++) {
                        a=-i*(nm1+i);
                        b+=2.0f;
                        d=1.0f/(a*d+b);
                        c=b+a/c;
                        del=c*d;
                        h*=del;
                        if (fabs(del-1.0f)<=epsilon) {
                                ans=h*exp(-x);
                                A[nm1*num_samples+idx] = ans;
                                return;
                        }
                }
        }
}
