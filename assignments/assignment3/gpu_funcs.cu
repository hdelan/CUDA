
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

//#include "gpu_funcs.h"

extern __constant__ double C_eulerConstant;

using namespace std;

__global__ void GPU_exponentialIntegralDouble_1 (const double start, const double end, const int num_samples, double division, double * A) {
        __shared__ int n; n=blockIdx.x+1;
        __shared__ double eulerConstant; eulerConstant=C_eulerConstant; //0.5772156649015329;
        __shared__ double psi;
        __shared__ double epsilon; epsilon=1.E-30;
        __shared__ double bigDouble; bigDouble=1.E100;
        int i,ii,nm1=n-1;
        double x,a=start,b=end,c,d,del,fact,h,ans=0.0;
        auto idx {threadIdx.x}, step {blockDim.x};

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

__global__ void GPU_exponentialIntegralDouble_2 (const double start, const double end, const int num_samples, const int max_n, double division, double * A) {
        int n=threadIdx.x+1;
        __shared__ double eulerConstant; eulerConstant=0.5772156649015329;
        __shared__ double psi;
        __shared__ double epsilon; epsilon=1.E-30;
        __shared__ double bigDouble; bigDouble=1.E100;
        int i,ii,nm1=n-1;
        double x=start+blockIdx.x*division,a=start,b=end,c,d,del,fact,h,ans=0.0;
        auto idx {threadIdx.x}, step {blockDim.x};

        if (n==0) {
                while (idx < num_samples) {
                        x = a+division*idx;
                        A[idx] = exp(-x)/x;
                        idx += step;
                }
        } else {
                if (x<=1.0) {
                        while (nm1 < max_n) {
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
                                                //A[nm1*num_samples+blockIdx.x] = ans;
                                                A[nm1+num_samples*blockIdx.x] = ans;
                                                n+=step;
                                                nm1+=step;
                                                break;
                                        }
                                }
                                if (i==2000000) printf("Series failed in exponential integral");
                        }
                }
                else {
                        while (nm1 < max_n) {
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
                                                //A[nm1*num_samples+blockIdx.x] = ans;
                                                A[nm1+num_samples*blockIdx.x] = ans;
                                                n+=step;
                                                nm1+=step;
                                                break;
                                        }
                                }
                        }
                }
        }
}

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

__global__ void GPU_exponentialIntegralDouble_4 (const double start, const double end, const int num_samples, const int max_n, double division, double * A) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        int n=idx+1;
        double x=(idy)*division+start;
        /*
        __shared__ double eulerConstant; eulerConstant=0.5772156649015329;
        __shared__ double psi;
        __shared__ double epsilon; epsilon=1.E-30;
        __shared__ double bigDouble; bigDouble=1.E100;
        */
        double eulerConstant=0.5772156649015329;
        double psi;
        double epsilon=1.E-30;
        double bigDouble=1.E300;
        int i,ii,nm1=n-1;
        double a=start,b=end,c,d,del,fact,h,ans=0.0;

        if (idy >= num_samples || idx >= max_n) return;

        if (n==0) {
                A[idx*num_samples+idy] = exp(-x)/x;
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
                                        A[idx*num_samples+idy] = ans;
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
                                        A[idx*num_samples+idy] = ans;
                                        return;
                                }
                        }
                }
        }
}


__global__ void GPU_exponentialIntegralDouble_5_launch (const double start, const double end, const int num_samples, const int max_n, double division, double * A) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        int n=idx+1;
        double x=(idy)*division+start;
        /*
        __shared__ double eulerConstant; eulerConstant=0.5772156649015329;
        __shared__ double psi;
        __shared__ double epsilon; epsilon=1.E-30;
        __shared__ double bigDouble; bigDouble=1.E100;
        */
        double eulerConstant=0.5772156649015329;
        double psi;
        double epsilon=1.E-30;
        double bigDouble=1.E300;
        int i,ii,nm1=n-1;
        double a=start,b=end,c,d,del,fact,h,ans=0.0;

        if (idy >= num_samples || idx >= max_n) return;

        if (n==0) {
                A[idx*num_samples+idy] = exp(-x)/x;
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
                                        A[idx*num_samples+idy] = ans;
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
                                        A[idx*num_samples+idy] = ans;
                                        return;
                                }
                        }
                }
        }
}
