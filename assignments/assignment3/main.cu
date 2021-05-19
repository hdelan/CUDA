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

#define BLOCK_SIZE 32

using namespace std;
__global__ void GPU_exponentialIntegralDouble_1 (const double start, const double end, const int num_samples, double division, double * A);

int main(int argc, char *argv[]) {
        unsigned int ui,uj;
        cpu=true, gpu=true;
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
                dim3 blocks {n}, threads{BLOCK_SIZE};
                double *A_d;
                cudaMalloc((void **) &A_d, sizeof(double)*numberOfSamples*n);

                if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
                        cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
                        exit(1);
                }

                GPU_exponentialIntegralDouble_1<<<blocks, threads>>>(a+division, b, numberOfSamples, division, A_d);

                double * resultsDoubleGpu {(double*)malloc(sizeof(double)*numberOfSamples*n)};
                cudaMemcpy(resultsDoubleGpu, A_d, sizeof(double)*numberOfSamples*n, cudaMemcpyDeviceToHost);

                printf("GPU version:\n\n");
                print_matrix_CPU(resultsDoubleGpu, n, numberOfSamples);

                printf("\n\nCPU version:\n\n");
                print_matrix_CPU(resultsDoubleCpu, n, numberOfSamples);
                
                std::cout << "=====>Error Checking" << std::endl;
                diff_matrices(resultsDoubleGpu, resultsDoubleCpu, n, numberOfSamples);

        }


        return 0;
}

