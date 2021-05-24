/**
 * \file:        gpu_funcs.cu
 
 * \brief:       Some gpu functions to calculate the exponential integral function.
                 
                 Uses GPU_exponentialIntegralDouble3 as a base. 
                 
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
// extern __constant__ double C_eulerConstant; 
// extern __constant__ double epsilon;
// extern __constant__ double bigDouble;

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
__global__ void GPU_exponentialIntegralDouble_3 (const double start, const double end, const int num_samples, const int start_n, const int max_n, double division, double * A) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        int n=idy+start_n;
        double x=(idx)*division+start;
        __shared__ double eulerConstant; eulerConstant=0.5772156649015329;
        __shared__ double psi;
        __shared__ double epsilon; epsilon=1.E-30;
        __shared__ double bigDouble; bigDouble=1.E100;
        int i,ii,nm1=n-1;
        double a=start,b=end,c,d,del,fact,h,ans=0.0;
        int dev;
        cudaGetDevice(&dev);
        //if (blockIdx.x+threadIdx.x==0) printf("Running for device %d\n", dev);
        
        if (idx >= num_samples || n >= max_n) return;

        if (n==0) {
                A[idy*num_samples+idx] = exp(-x)/x;
                return;
        } else {
                if (x<=1.0) {
                        ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
                        fact=1.0;
                        for (i=1;i<=20000000;i++) {
                                fact*=-x/i;
                                if (i != nm1) {
                                        del = -fact/(i-nm1);
                                } else {
                                        psi = -eulerConstant;
                                        for (ii=1;ii<=nm1;ii++) {
                                                psi += 1.0/ii;
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
                        c=bigDouble;
                        d=1.0/b;
                        h=d;
                        for (i=1;i<=20000000;i++) {
                                a=-i*(nm1+i);
                                b+=2.0;
                                d=1.0/(a*d+b);
                                c=b+a/c;
                                del=c*d;
                                h*=del;
                                if (fabs(del-1.0)<=epsilon) {
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
 * \param:       resultsDoubleGpu
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
void launch_on_two_cards(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned block_size, const unsigned yblock, float & time_taken) {

  // Create streams
  cudaStream_t stream[2];


  int count;
  cudaGetDeviceCount(&count);
  printf("Number of graphics cards: %d\n", count);

  dim3 blocks {numberOfSamples/block_size+1, n/(2*yblock)+1}, threads {block_size, yblock};

  double *A_d0, *A_d1;

  /*************** CARD 1 **************/
  cudaSetDevice(0);
  cudaStreamCreate(&stream[0]);

  // TIMINGS
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, stream[0]); // We use 0 here because it is the "default" stream

  cudaMalloc((void **) &A_d0, sizeof(double)*numberOfSamples*(n/2));
  GPU_exponentialIntegralDouble_3<<<blocks, threads, 0, stream[0]>>>(a+division, b, numberOfSamples,1,n/2+1, division,A_d0);
  cudaMemcpyAsync(resultsDoubleGpu, A_d0, sizeof(double)*numberOfSamples*n/2, cudaMemcpyDeviceToHost, stream[0]);

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

  cudaMalloc((void **) &A_d1, sizeof(double)*numberOfSamples*(n/2 + (!(n%2)?0:1)));
  GPU_exponentialIntegralDouble_3<<<blocks, threads,0,stream[1]>>>(a+division, b, numberOfSamples,n/2+1,n+1,division,A_d1);
  cudaMemcpyAsync(resultsDoubleGpu+(numberOfSamples)*n/2, A_d1, sizeof(double)*numberOfSamples*(n/2 + (!(n%2)?0:1)), cudaMemcpyDeviceToHost, stream[1]);

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
 * \param:       resultsDoubleGpu       output
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
void launch_on_one_card(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned block_size, const unsigned yblock, float & time_taken) {
  // TIMINGS
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, 0); // We use 0 here because it is the "default" stream
  
  dim3 blocks3 {numberOfSamples/block_size+1, n/yblock+1}, threads3 {block_size, yblock};
  double *A_d;
  cudaMalloc((void **) &A_d, sizeof(double)*numberOfSamples*n);

  // RUN KERNEL
  GPU_exponentialIntegralDouble_3<<<blocks3, threads3>>>(a+division, b, numberOfSamples, 1, n+1, division, A_d);

  cudaMemcpy(resultsDoubleGpu, A_d, sizeof(double)*numberOfSamples*n, cudaMemcpyDeviceToHost);

  // TIMINGS
  cudaEventRecord(computeFloatGpuEnd, 0);
  cudaEventSynchronize(computeFloatGpuStart);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime, computeFloatGpuStart, computeFloatGpuEnd);
  computeFloatGpuTime=(float)(computeFloatGpuElapsedTime)*0.001;
  
  if (timing) printf("Total GPU timings: %E seconds\n", computeFloatGpuTime);
  time_taken = computeFloatGpuTime;
}
