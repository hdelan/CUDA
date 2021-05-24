/**
 * \file:        main.cu
 * \brief:       Main file to calculate exponential integral function in serial
                 and on GPU
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-05-24
 */

#include "cpu_funcs.h"
#include "gpu_funcs.h"

using namespace std;

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

  double * resultsDoubleCpu;

  double timeTotalCpu=0.0;

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



    if (n<=100&&numberOfSamples<=100) printf("GPU version:\n\n");
    print_matrix_CPU(resultsDoubleGpu, n, numberOfSamples);

    if (n<=100&&numberOfSamples<=100) printf("\n\nCPU version:\n\n");
    print_matrix_CPU(resultsDoubleCpu, n, numberOfSamples);

    if (cpu) {
      std::cout << "\n=====>Error Checking" << std::endl;
      diff_matrices(resultsDoubleGpu, resultsDoubleCpu, n, numberOfSamples);
      std::cout << "\n=====>Timings (seconds)" << std::endl;

      std::cout << "\tCPU:\t\t" << timeTotalCpu << std::endl;
      std::cout << "\tGPU:\t\t" << gpu_time << std::endl;
      std::cout << "\tSpeedup:\t" << timeTotalCpu/gpu_time << std::endl;
    }

  }

  return 0;
}

