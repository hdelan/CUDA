
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

#define BLOCK_SIZE 32

using namespace std;
__global__ void GPU_exponentialIntegralDouble_1 (const double start, const double end, const int num_samples, double division, double * A);


__global__ void GPU_exponentialIntegralDouble_1 (const double start, const double end, const int num_samples, double division, double * A) {
        int n=blockIdx.x+1;
        /*constant*/ const double eulerConstant=0.5772156649015329;
        /*constant*/ double epsilon=1.E-30;
        /*constant*/ double bigDouble=1.E100;
        int i,ii,nm1=n-1;
        double x,a=start,b=end,c,d,del,fact,h,psi,ans=0.0;
        auto glob_idx {blockIdx.x*n+threadIdx.x}, idx {threadIdx.x}, step {blockDim.x};

        if (n==0) {
                while (idx < num_samples) {
                        x = a+division*idx;
                        A[idx] = exp(-x)/x;
                        idx += step;
                }
        } else {
                x = a+division*idx;
                //if (threadIdx.x+blockIdx.x == 1) printf("0Hello from thread %d block %d a=%lf x=%lf step=%d\n", threadIdx.x, blockIdx.x, a, x, step);
                while (x<=1.0) {
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
                                        //printf("Writing for idx: %d\n", idx);
                                        A[blockIdx.x*num_samples+idx] = ans;
                                        idx += step;
                                        x = a+division*idx;
                                        break;
                                }
                        }
                        if (i==2000000) printf("Series failed in exponential integral");
                }
                //if (threadIdx.x+blockIdx.x == 1) printf("Hello from thread %d block %d a=%lf x=%lf\n", threadIdx.x, blockIdx.x, a, x);
                //cout << "Series failed in exponentialIntegral" << endl;
                while (idx < num_samples) {
                        //if (threadIdx.x+blockIdx.x == 1) printf("Hello from thread %d block %d a=%lf x=%lf\n", threadIdx.x, blockIdx.x, a, x);
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
                                        //if (blockIdx.x*num_samples+idx==2) printf("Hello from thread %d block %d of %d a=%lf x=%lf h=%lf ans=%lf a=%lf b=%lf c=%lf d=%lf del=%E step=%u\n", threadIdx.x, blockIdx.x, blockDim.x,a, x, h, ans, a,b,c,d,del,step);
                                        A[blockIdx.x*num_samples+idx] = ans;
                                        idx+=blockDim.x;
                                        x=start+division*idx;
                                        break;
                                }
                        }
                }
                //cout << "Continued fraction failed in exponentialIntegral" << endl;
        }
}

