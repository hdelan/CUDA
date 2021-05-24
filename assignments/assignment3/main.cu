///// Created by Jose Mauricio Refojo - 2014-04-02		Last changed: 2017-04-05
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>

#include "cpu_funcs.h"
#include "gpu_funcs.h"

using namespace std;

void launch_on_two_cards(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned, const unsigned, float &);
void launch_on_one_card(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned block_size, const unsigned yblock, float &);

bool cpu=true, gpu=true, verbose=false, timing=true, split=false;
double a=0.0;
double b=10.0;
double division;
unsigned numberOfSamples=10;
unsigned n=10;
unsigned block_size=32, yblock;
int maxIterations=2000000000;

__constant__ double C_eulerConstant;

int main(int argc, char *argv[]) {
  unsigned int ui,uj;
  
  float gpu_time;

  struct timeval expoStart, expoEnd;
  
  parseArguments (argc, argv);

  unsigned yblock {1024/block_size};

  if (verbose) {
    cout << "n=" << n << endl;
    cout << "numberOfSamples=" << numberOfSamples << endl;
    cout << "a=" << a << endl;
    cout << "b=" << b << endl;
    cout << "timing=" << timing << endl;
    cout << "verbose=" << verbose << endl;
    cout << "x_block_size=" << block_size << endl;
    cout << "y_block_size=" << yblock << endl;
  }

  // Sanity checks
  if (a>=b) {
    cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
    return 0;
  }
  if (n<=0) {
    cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
    return 0;
  }
  if (numberOfSamples<=0) {
    cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
    return 0;
  }

  float * resultsFloatCpu;
  double * resultsDoubleCpu;

  double timeTotalCpu=0.0;

  try {
    resultsFloatCpu = (float *) malloc(sizeof(float)*numberOfSamples*n);
  } catch (std::bad_alloc const&) {
    cout << "resultsFloatCpu memory allocation fail!" << endl;	exit(1);
  }
  try {
    resultsDoubleCpu = (double *) malloc(sizeof(double)*numberOfSamples*n);
  } catch (std::bad_alloc const&) {
    cout << "resultsDoubleCpu memory allocation fail!" << endl;	exit(1);
  }

  double x,division=(b-a)/((double)(numberOfSamples));

  if (cpu) {
    gettimeofday(&expoStart, NULL);
    for (ui=1;ui<=n;ui++) {
      for (uj=1;uj<=numberOfSamples;uj++) {
        x=a+uj*division;
        resultsFloatCpu[(ui-1)*numberOfSamples+uj-1]=exponentialIntegralFloat (ui,x);
        resultsDoubleCpu[(ui-1)*numberOfSamples+uj-1]=exponentialIntegralDouble (ui,x);
      }
    }
    gettimeofday(&expoEnd, NULL);
    timeTotalCpu=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
  }

  if (timing) {
    if (cpu) {
      printf ("calculating the exponentials on the cpu took: %f seconds\n",timeTotalCpu);
    }
  }

  if (verbose) {
    if (cpu) {
      //outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
    }
  }
  
  // GPU EXECUTION
  if (gpu) {

    double * resultsDoubleGpu;
    try {
      resultsDoubleGpu = (double *) malloc(sizeof(double)*numberOfSamples*n);
    } catch (std::bad_alloc const&) {
      cout << "resultsDoubleGpu memory allocation fail!" << endl;	exit(1);
    }
    if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
      cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
      exit(1);
    }
    if (split) {
      /*************** USE 2 GPUS **************/
      launch_on_two_cards(resultsDoubleGpu, n, numberOfSamples, a, b, division, block_size, yblock, gpu_time);

    } else { 
      /*************** USE 1 GPU **************/
      launch_on_one_card(resultsDoubleGpu, n, numberOfSamples, a, b, division, block_size, yblock, gpu_time);

    }



    printf("GPU version:\n\n");
    print_matrix_CPU(resultsDoubleGpu, n, numberOfSamples);

    printf("\n\nCPU version:\n\n");
    print_matrix_CPU(resultsDoubleCpu, n, numberOfSamples);

    if (cpu) {
      std::cout << "=====>Error Checking" << std::endl;
      diff_matrices(resultsDoubleGpu, resultsDoubleCpu, n, numberOfSamples);
      std::cout << "\n=====>Timings (seconds)" << std::endl;

      std::cout << "\tCPU:\t\t" << timeTotalCpu << std::endl;
      std::cout << "\tGPU:\t\t" << gpu_time << std::endl;
      std::cout << "\tSpeedup:\t" << timeTotalCpu/gpu_time << std::endl;
    }

  }


  return 0;
}

// We are going to split the 1d array in half which corresponds to a horizontal split,
// so one card will calculate the entire interval for n=1 to n=n/2 and another will calculate
// for n=n/2 to n=n

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


void launch_on_one_card(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned block_size, const unsigned yblock, float & time_taken) {
  // TIMINGS
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, 0); // We use 0 here because it is the "default" stream


  double euler=0.5772156649015329;
  cudaMemcpyToSymbol(C_eulerConstant, &euler, sizeof(double));
  dim3 blocks1 {n}, blocks2 {numberOfSamples}, threads {block_size}; 
  dim3 blocks3 {numberOfSamples/block_size+1, n/yblock+1}, threads3 {block_size, yblock};
  double *A_d;
  cudaMalloc((void **) &A_d, sizeof(double)*numberOfSamples*n);
  //GPU_exponentialIntegralDouble_2<<<blocks2, threads>>>(a+division, b, numberOfSamples, n, division, A_d);
  //GPU_exponentialIntegralDouble_1<<<blocks1, threads>>>(a+division, b, numberOfSamples, division, A_d);
  GPU_exponentialIntegralDouble_3<<<blocks3, threads3>>>(a+division, b, numberOfSamples, 1, n+1, division, A_d);
  //GPU_exponentialIntegralDouble_4<<<blocks3, threads3>>>(a+division, b, numberOfSamples, n, division, A_d);

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
