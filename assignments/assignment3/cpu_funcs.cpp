///// Created by Jose Mauricio Refojo - 2014-04-02		Last changed: 2017-04-05
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include "cpu_funcs.h"

using namespace std;

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

void diff_matrices(float *A, float *C, unsigned int n, unsigned int m) {
        unsigned int index_r=0, index_c=0, count=0, gpu_bigger=0;
        float max_diff = 0.0f, diff = 0.0f;
        for (auto i=0u;i<n*m;i++) {
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


void diff_matrices(double *A, double *C, unsigned int n, unsigned int m) {
        unsigned int index_r=0, index_c=0, count=0, gpu_bigger=0;
        double max_diff = 0.0f, diff = 0.0f;
        for (auto i=0u;i<n*m;i++) {
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

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to print a matrix to stdout if it is smaller than 100 x 100
 *
 * \param:       A
 * \param:       N
 * \param:       M
 */
/* ----------------------------------------------------------------------------*/
void print_matrix_CPU(double * A, const unsigned int N, const unsigned int M) {
        if (N > 100 || M > 100) {
                return;
        }	

        for (auto i = 0u; i < N; i++) {
                std::cout << " | ";
                for (auto j = 0u; j < M; j++) 
                        std::cout << std::setw(9) << std::setprecision(2)  << A[i*M + j];
                std::cout << " |\n";
        }
        std::cout << "\n";
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Jose Refojo's exponential integral function to be calculated in 
 *               serial
 *
 * \param:       n
 * \param:       x
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
double exponentialIntegralDouble (const int n,const double x) {
        static const double eulerConstant=0.5772156649015329;
        double epsilon=1.E-30;
        double bigDouble=std::numeric_limits<double>::max();
        int i,ii,nm1=n-1;
        double a,b,c,d,del,fact,h,psi,ans=0.0;


        if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
                cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
                exit(1);
        }
        if (n==0) {
                ans=exp(-x)/x;
        } else {
                if (x>1.0) {
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
                                        return ans;
                                }
                        }
                        //cout << "Continued fraction failed in exponentialIntegral" << endl;
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
                                        //std::cout << "Called!\n";
                                        for (ii=1;ii<=nm1;ii++) {
                                                psi += 1.0/ii;
                                        }
                                        del=fact*(-log(x)+psi);
                                }
                                ans+=del;
                                if (fabs(del)<fabs(ans)*epsilon) return ans;
                        }
                        //cout << "Series failed in exponentialIntegral" << endl;
                        return ans;
                }
        }
        return ans;
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Jose Refojo's exponential integral function to be calculated in 
 *               serial
 *
 * \param:       n
 * \param:       x
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
float exponentialIntegralFloat (const int n,const float x) {
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
                return exp(-x)/x;
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

        while ((c = getopt (argc, argv, "cshn:m:a:b:tv")) != -1) {
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
                        case 's':
                                split = true; break;
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
