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

#define BLOCK_SIZE 32

using namespace std;

float	exponentialIntegralFloat		(const int n,const float x);
double	exponentialIntegralDouble		(const int n,const double x);
void	outputResultsCpu			(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int		parseArguments				(int argc, char **argv);
void	printUsage				(void);


bool verbose,timing,cpu;
int maxIterations;
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use

int main(int argc, char *argv[]) {
  unsigned int ui,uj;
  cpu=true;
  verbose=false;
  timing=true;
  // n is the maximum order of the exponential integral that we are going to test
  // numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
  n=10;
  numberOfSamples=10;
  a=0.0;
  b=10.0;
  maxIterations=2000000000;

  struct timeval expoStart, expoEnd;

  parseArguments(argc, argv);

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

  std::vector< std::vector< float  > > resultsFloatCpu;
  std::vector< std::vector< double > > resultsDoubleCpu;

  double timeTotalCpu=0.0;

  try {
    resultsFloatCpu.resize(n,vector< float >(numberOfSamples));
  } catch (std::bad_alloc const&) {
    cout << "resultsFloatCpu memory allocation fail!" << endl;	exit(1);
  }
  try {
    resultsDoubleCpu.resize(n,vector< double >(numberOfSamples));
  } catch (std::bad_alloc const&) {
    cout << "resultsDoubleCpu memory allocation fail!" << endl;	exit(1);
  }

  double x,division=(b-a)/((double)(numberOfSamples));

  if (cpu) {
    gettimeofday(&expoStart, NULL);
    for (ui=1;ui<=n;ui++) {
      for (uj=1;uj<=numberOfSamples;uj++) {
        x=a+uj*division;
        resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat (ui,x);
        resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble (ui,x);
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
      outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
    }
  }

  if (gpu) {
    dim3 blocks {n}, threads{BLOCK_SIZE};
    float * x_d;
    cudaMalloc((void **) &x_d, sizeof(float)*numberOfSamples*n);
    
    if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
      cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
      exit(1);
    }

    GPU_exponentialIntegralDouble_1<<<blocks, threads>>>(a, b, numberOfSamples, division, x_d);

  }


  return 0;
}

__global__ double GPU_exponentialIntegralDouble_1 (const double a, const double b, const int num_samples, double division, double * A) {
  /*constant*/ const double eulerConstant=0.5772156649015329;
  /*constant*/ double epsilon=1.E-30;
  /*constant*/ double bigDouble=std::numeric_limits<double>::max();
  int i,ii,nm1=n-1;
  double x,a,b,c,d,del,fact,h,psi,ans=0.0;
  int glob_idx {blockIdx.x*n+threadIdx.x}, idx {threadIdx.x}, step {blockDim.x};

  if (blockIdx.x==0) {
    while (idx < num_samples) {
      x = a+division*idx;
      A[idx] = exp(-x)/x;
      idx += step;
    }
  } else {
    x = a+division*idx;
    while (x<=1.0) {
      ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
      fact=1.0;
      for (i=1;i<=maxIterations;i++) {
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
        if (fabs(del)<fabs(ans)*epsilon) A[blockIdx.x*num_sample+idx] = ans;
      }
      idx += step;
      x = a+division*idx;
    }
    //cout << "Series failed in exponentialIntegral" << endl;
    while (idx < num_samples) {
      b=x+n;
      c=bigDouble;
      d=1.0/b;
      h=d;
      for (i=1;i<=maxIterations;i++) {
        a=-i*(nm1+i);
        b+=2.0;
        d=1.0/(a*d+b);
        c=b+a/c;
        del=c*d;
        h*=del;
        if (fabs(del-1.0)<=epsilon) {
          ans=h*exp(-x);
          A[blockIdx.x*num_samples+idx] = ans;
        }
      }
      idx += step;
      x = a+division*idx;
    }
    //cout << "Continued fraction failed in exponentialIntegral" << endl;
  }
}

__global__ float exponentialIntegralFloat (const int n,const float x) {
  static const float eulerConstant=0.5772156649015329;
  float epsilon=1.E-30;
  float bigfloat=std::numeric_limits<float>::max();
  int i,ii,nm1=n-1;
  float a,b,c,d,del,fact,h,psi,ans=0.0;

  if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
    cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
    exit(1);
  }
  if (n==0) {
    ans=exp(-x)/x;
  } else {
    if (x>1.0) {
      b=x+n;
      c=bigfloat;
      d=1.0/b;
      h=d;
      for (i=1;i<=maxIterations;i++) {
        a=-i*(nm1+i);
        b+=2.0;
        d=1.0/(a*d+b);
        c=b+a/c;
        del=c*d;
        h*=del;
        if (fabs(del-1.0)<=epsilon) {
          ans=h*exp(-x);
          return ans;
        }
      }
      ans=h*exp(-x);
      return ans;
    } else { // Evaluate series
      ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
      fact=1.0;
      for (i=1;i<=maxIterations;i++) {
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
        if (fabs(del)<fabs(ans)*epsilon) return ans;
      }
      return ans;
    }
  }
  return ans;
}


int parseArguments (int argc, char *argv[]) {
  int c;

  while ((c = getopt (argc, argv, "cghn:m:a:b:tv")) != -1) {
    switch(c) {
      case 'c':
        cpu=false; break;	 //Skip the CPU test
      case 'h':
        printUsage(); exit(0); break;
      case 'i':
        maxIterations = atoi(optarg); break;
      case 'n':
        n = atoi(optarg); break;
      case 'm':
        numberOfSamples = atoi(optarg); break;
      case 'a':
        a = atof(optarg); break;
      case 'b':
        b = atof(optarg); break;
      case 't':
        timing = true; break;
      case 'v':
        verbose = true; break;
      default:
        fprintf(stderr, "Invalid option given\n");
        printUsage();
        return -1;
    }
  }
  return 0;
}
void printUsage () {
  printf("exponentialIntegral program\n");
  printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
  printf("This program will calculate a number of exponential integrals\n");
  printf("usage:\n");
  printf("exponentialIntegral.out [options]\n");
  printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
  printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
  printf("      -c           : will skip the CPU test\n");
  printf("      -g           : will skip the GPU test\n");
  printf("      -h           : will show this usage\n");
  printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
  printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
  printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
  printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
  printf("      -v           : will activate the verbose mode  (default: no)\n");
  printf("     \n");
}
/**
 * \file:        main.cu
 * \brief:       A main function to calculate the cylindrical radiator finite differences
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-04-29
 */

#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <cuda_profiler_api.h> 

#include "blockdim.h"
#include "cpu_funcs.hpp"
#include "gpu_funcs.h"

/*
__global__ void gpu_rad_sweep5(float*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_rad_sweep6(float*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_get_averages(float * a, unsigned int n, unsigned int m, float * avg);
void print_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M);
void read_matrix_from_file(std::string filename, float * A);

void print_sparse_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M, const unsigned int iters);
void read_sparse_matrix_from_file(std::string filename, float * A);

template <typename T>
void cpu_rad_sweep1(T*, unsigned int, unsigned int, unsigned int, T*);
template <typename T>
void cpu_rad_sweep2(T*, unsigned int, unsigned int, unsigned int, T*);
void get_averages(float * a, unsigned int n, unsigned int m, float * avg);

void diff_matrices(float *A, float *B, unsigned int n, unsigned int m);

void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, int & cpu_calc, unsigned int & block_size, int & write_file);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);
*/
int main(int argc, char * argv[]) { 
  unsigned int n {15360}, m {15360}, block_size {BLOCK_SIZE}, num_iters {500};
  long unsigned int seed {123};
  int print_time {0}, cpu_calc {1}, diff_mats {1}, write_mat {0};
  struct timeval t1, t2, t3, t4;
  parse_command_line(argc, argv, n, m, num_iters, seed, print_time, cpu_calc, block_size, write_mat);

  std::cout << "n: " << n << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "iters: " << num_iters << "\n";
  std::cout << "print_time: " << print_time << "\n";
  std::cout << "block_size: " << block_size << "\n";
  // A 
  float * A = (float *) calloc(n*m, sizeof(float));
  float * C = (float *) calloc(n*m, sizeof(float));

  // Set boundary conditions
  for (auto i=0;i<n;++i) {
    A[i*m] = C[i*m] = 1.0f*(float)(i+1)/(float)n;
    A[i*m+1] = C[i*m+1] = 0.80f*(float)(i+1)/(float)n;
  }

  float *A_d, *avg_d;
  cudaMalloc((void **) &A_d, sizeof(float)*n*m);
  cudaMalloc((void **) &avg_d, sizeof(float)*n);
  cudaMemcpy(A_d, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
  
  // Initialize block_size
  dim3 threads {block_size};
  dim3 blocks {(n/threads.x) + (!(n%threads.x)?0:1)};
  dim3 n_blocks {n};

  std::cout << "Num blocks: " << n_blocks.x << '\n';


  /************ CPU FUNCTION/READ IN FILE *************/
  // This file will be written to if the -c flag is not provided, or it will be read from
  std::string sparse_filename {"data/sp_CPU" + std::to_string(n) + "x" + std::to_string(m) + "_p" + std::to_string(num_iters) + ".txt"};

  if (cpu_calc == 1) {
    gettimeofday(&t1, NULL);
    cpu_rad_sweep1(C, n, m, num_iters, A);
    gettimeofday(&t2, NULL);
    // Every time the CPU code runs it will save the matrix to file
    print_sparse_matrix_to_file(sparse_filename, C, n, m, num_iters);
  } else {
    try {
      std::cout << "Reading SPARSE CPU-generated comparison matrix from file.\n";
      gettimeofday(&t1, NULL);
      read_sparse_matrix_from_file(sparse_filename, C);
      // Uncomment this to read a non-sparse matrix from file
      //read_matrix_from_file(other_filename, C);
      gettimeofday(&t2, NULL);
      printf("Finished reading sparse matrix from file. Time taken: %lf\n", (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0));
    } catch (std::exception e) {
      std::cout << "Could not read from file! Will not be able to compare GPU matrix with CPU matrix.\n";
      std::cout << "Caught " << e.what() << "\n";
      diff_mats = 0;
    }
  }

  /************ RUNNING KERNELS *************/
  cudaEvent_t computeFloatGpuStart, computeFloatGpuEnd;
  float computeFloatGpuElapsedTime,computeFloatGpuTime;
  cudaEventCreate(&computeFloatGpuStart);
  cudaEventCreate(&computeFloatGpuEnd);
  cudaEventRecord(computeFloatGpuStart, 0); // We use 0 here because it is the "default" stream
  
  // gpu_rad_sweep6 should only be used when there are zeros in the middle of matrices,
  // otherwise gpu_rad_sweep5 is better. See gpu_funcs.cu
  if (8*(num_iters+2) < m && 8*(num_iters+2) < 12288) {
    gpu_rad_sweep6<<<n_blocks, threads, 4*8*(num_iters+2)>>>(A_d, n, m, num_iters);
  } else {
    gpu_rad_sweep5<<<n_blocks, threads>>>(A_d, n, m, num_iters);
  }
  
  cudaEventRecord(computeFloatGpuEnd, 0);
  cudaEventSynchronize(computeFloatGpuStart);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime, computeFloatGpuStart, computeFloatGpuEnd);
  computeFloatGpuTime=(float)(computeFloatGpuElapsedTime)*0.001;

  // Transfer to RAM
  cudaMemcpy(A, A_d, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

  // Print GPU_mat to file?
  if (write_mat == 1) print_sparse_matrix_to_file("data/sp_GPU" + std::to_string(n) + "x" + std::to_string(m) + "_p" + std::to_string(num_iters), A, n, m, num_iters);


  /************ CALCULATE AVERAGES *************/
  float avgC[n], avgA[n], avgD[n];
  
  // CPU avgs of GPU function
  gettimeofday(&t3, NULL);
  get_averages(A, n, m, avgA);
  gettimeofday(&t4, NULL);
  
  // GPU avgs of GPU function
  cudaEvent_t computeFloatGpuStart1, computeFloatGpuEnd1;
  float computeFloatGpuElapsedTime1,computeFloatGpuTime1;
  cudaEventCreate(&computeFloatGpuStart1);
  cudaEventCreate(&computeFloatGpuEnd1);
  cudaEventRecord(computeFloatGpuStart1, 0); 
  
  gpu_get_averages<<<blocks, threads>>>(A_d, n, m, avg_d);
  
  cudaEventRecord(computeFloatGpuEnd1, 0);
  cudaEventSynchronize(computeFloatGpuStart1);  // This is optional, we shouldn't need it
  cudaEventSynchronize(computeFloatGpuEnd1); // This isn't - we need to wait for the event to finish
  cudaEventElapsedTime(&computeFloatGpuElapsedTime1, computeFloatGpuStart1, computeFloatGpuEnd1);
  computeFloatGpuTime1=(float)(computeFloatGpuElapsedTime1)*0.001;
  
  cudaMemcpy(avgD, avg_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
  
  
  /************ COMPARE RESULTS ****************/
  std::cout << "\n=====>Errors\n";
  std::cout << "   Of matrix elements: \n";
  // Diff matrices - this will be avoided if there was an error reading in the comparison file
  if (diff_mats == 1) diff_matrices(A, C, n, m);
  std::cout << "   Of GPU_avg function: \n";
  if (diff_mats == 1) diff_matrices(avgD, avgA, n, 1);
  get_averages(C, n, m, avgC);
  std::cout << "   Of averages (CPU vs GPU): \n";
  if (diff_mats == 1) diff_matrices(avgA, avgC, n, 1);
  

  /************ PRINT TIMINGS ******************/
  double cpu_rad_time, cpu_avgs_time;

  if (print_time == 1) {
    if (n == 15360 && m == 15360 && num_iters == 500) {
      // This value is stored in the file cpu_timing_p500
      cpu_rad_time = 353.621;
    } else if (cpu_calc == 1) {
      cpu_rad_time = (double)(t2.tv_sec-t1.tv_sec)+((double)(t2.tv_usec - t1.tv_usec)/1000000.0);
    } else { 
      return 0;
    }

    cpu_avgs_time = (double)(t4.tv_sec-t3.tv_sec)+((double)(t4.tv_usec - t3.tv_usec)/1000000.0);
    printf("\n=====>Radiator timings (block size %d):\n\tCPU_timing: \t%lf\n\tGPU_timing: \t%lf\n\tSpeedup: \t%lf\n\n", BLOCK_SIZE, cpu_rad_time, computeFloatGpuTime, cpu_rad_time/computeFloatGpuTime);
    printf("\n=====>Row average timings (block size %d):\n\tCPU_timing: \t%lf\n\tGPU_timing: \t%lf\n\tSpeedup: \t%lf\n\n", BLOCK_SIZE, cpu_avgs_time, computeFloatGpuTime1, cpu_avgs_time/computeFloatGpuTime1);
  }
  /*
     cudaError_t err = cudaGetLastError();  // add
     if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
     cudaProfilerStop();
   */

  free(A); free(C);
  cudaFree(A_d);

  return 0;
}

void diff_matrices(float *A, float *C, unsigned int n, unsigned int m) {
  unsigned int index_r=0, index_c=0, count=0, gpu_bigger=0;
  float max_diff = 0.0f, diff = 0.0f;
  for (int i=0;i<n*m;i++) {
    diff = fabs(A[i] - C[i]);
    if (diff > 0.00001f) {
      if (A[i] > C[i]) gpu_bigger++;
      count++;
    }
    if (diff > max_diff) {
      max_diff = diff;
      index_r = i / m;
      index_c = i % m;
    }
  }
  std::cout << "\tDifference in entries greater than 1e-5 at " << count << " of " << n*m << " points\n";
  std::cout << "\tGPU bigger at " << gpu_bigger << " of " << count << " points.\n";
  std::cout << "\tMax diff: " << max_diff << " at index (" << index_r << ", " << index_c << ")\n";
  if (max_diff != 0.0f) {
    std::cout << "\tGPU_mat[i]: " << A[index_r*m+index_c] << "\n\tCPU_mat[i]: " << C[index_r*m+index_c] << "\n";
  }
}


