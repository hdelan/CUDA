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

#define BLOCK_SIZE 512

#define XBLOCK 512
#define YBLOCK 2

using namespace std;

bool cpu=true, gpu=true, verbose=false, timing=true, split=false;
double a=0.0;
double b=10.0;
unsigned numberOfSamples=10;
unsigned n=10;
int maxIterations=2000000000;

__constant__ double C_eulerConstant;

int main(int argc, char *argv[]) {
  unsigned int ui,uj;
 /* bool cpu=true, gpu=true, verbose=false, timing=true, split=false;
  // n is the maximum order of the exponential integral that we are going to test
  // numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
  unsigned n=10;
  unsigned numberOfSamples=10;
  double a=0.0;
  double b=10.0;
  int maxIterations=2000000000;*/

  struct timeval expoStart, expoEnd;

  //parseArguments (argc, argv, maxIterations, n, numberOfSamples, a, b, timing, verbose, cpu, split);
  parseArguments (argc, argv);

  if (verbose) {
    cout << "n=" << n << endl;
    cout << "numberOfSamples=" << numberOfSamples << endl;
    cout << "a=" << a << endl;
    cout << "b=" << b << endl;
    cout << "timing=" << timing << endl;
    cout << "verbose=" << verbose << endl;
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

  if (gpu) {

    double * resultsDoubleGpu {(double*)malloc(sizeof(double)*numberOfSamples*n)};
    if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
      cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
      exit(1);
    }
    if (split) {
      // We are going to split the 1d array in half which corresponds to a horizontal split,
      // so one card will calculate the entire interval for n=1 to n=n/2 and another will calculate
      // for n=n/2 to n=n

      // Create streams
      cudaStream_t stream[2];
      
      int count;
      cudaGetDeviceCount(&count);
      printf("Number of graphics cards: %d\n", count);

      dim3 blocks {numberOfSamples/XBLOCK+1, n/(2*YBLOCK)+1}, threads {XBLOCK, YBLOCK};
      
      double *A_d0, *A_d1;
      
      cudaSetDevice(0);
      cudaStreamCreate(&stream[0]);
      cudaMalloc((void **) &A_d0, sizeof(double)*numberOfSamples*(n/2));
      GPU_exponentialIntegralDouble_3<<<blocks, threads, 0, stream[0]>>>(a+division, b, numberOfSamples,1,n/2+1, division,A_d0);
      cudaMemcpyAsync(resultsDoubleGpu, A_d0, sizeof(double)*numberOfSamples*n/2, cudaMemcpyDeviceToHost, stream[0]);
      
      cudaSetDevice(1);
      cudaStreamCreate(&stream[1]);
      cudaMalloc((void **) &A_d1, sizeof(double)*numberOfSamples*(n/2 + (!(n%2)?0:1)));
      GPU_exponentialIntegralDouble_3<<<blocks, threads,0,stream[1]>>>(a+division, b, numberOfSamples,n/2+1,n+1,division,A_d1);
      cudaMemcpyAsync(resultsDoubleGpu+(numberOfSamples)*n/2, A_d1, sizeof(double)*numberOfSamples*(n/2 + (!(n%2)?0:1)), cudaMemcpyDeviceToHost, stream[1]);


    } else { 
      double euler=0.5772156649015329;
      cudaMemcpyToSymbol(C_eulerConstant, &euler, sizeof(double));
      dim3 blocks1 {n}, blocks2 {numberOfSamples}, threads {BLOCK_SIZE}; 
      dim3 blocks3 {numberOfSamples/XBLOCK+1, n/YBLOCK+1}, threads3 {XBLOCK, YBLOCK};
      double *A_d;
      cudaMalloc((void **) &A_d, sizeof(double)*numberOfSamples*n);
      GPU_exponentialIntegralDouble_2<<<blocks2, threads>>>(a+division, b, numberOfSamples, n, division, A_d);
      GPU_exponentialIntegralDouble_1<<<blocks1, threads>>>(a+division, b, numberOfSamples, division, A_d);
      GPU_exponentialIntegralDouble_3<<<blocks3, threads3>>>(a+division, b, numberOfSamples, 1, n+1, division, A_d);
      GPU_exponentialIntegralDouble_4<<<blocks3, threads3>>>(a+division, b, numberOfSamples, n, division, A_d);

      cudaMemcpy(resultsDoubleGpu, A_d, sizeof(double)*numberOfSamples*n, cudaMemcpyDeviceToHost);
    }

    printf("GPU version:\n\n");
    print_matrix_CPU(resultsDoubleGpu, n, numberOfSamples);

    printf("\n\nCPU version:\n\n");
    print_matrix_CPU(resultsDoubleCpu, n, numberOfSamples);

    if (cpu) {
      std::cout << "=====>Error Checking" << std::endl;
      diff_matrices(resultsDoubleGpu, resultsDoubleCpu, n, numberOfSamples);
    }

  }


  return 0;
}

