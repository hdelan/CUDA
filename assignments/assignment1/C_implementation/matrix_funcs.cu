#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#include "matrix.h"

unsigned int MAX_DIM = 1000000;

// KERNELS
__global__ void sum_abs_rows_GPU(float * data, float * rowsum, int N, int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        rowsum[idx] = 0.0;
        if (idx < N) {
                for (int j = 0; j < M; ++j) {
                        rowsum[idx] += std::abs(data[idx*M + j]);
                }
        }
}

__global__ void sum_abs_cols_GPU(float * data, float * colsum, int N, int M) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        colsum[idx] = 0.0;
        if (idx < N) {
                for (int j = 0; j < M; ++j) {
                        colsum[idx] += std::abs(data[idx + j*M]);
                }
        }
}

__global__ void reduce0_GPU(float * vector, float * alt_vector, int N) {
	int stride = blockDim.x * gridDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int index = idx;
	alt_vector[idx] = 0.0f;
	while (index < N) {
		alt_vector[idx] += vector[index];
		index += stride;
	}
}


__global__ float reduce1_GPU(float * alt_vector,  int N) {
	int stride = blockDim.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= stride) {
		return;
	}
	int index = idx + stride;
	while (index < N) {
		alt_vector[idx] += alt_vector[index];
		index += stride;
	}
}

void vector_reduction_GPU(float * vector, int N) {
	// stride is the number of active threads, which will reduce for each iteration
	// of the while loop
	float threadsum
	while (index < N){
		altvector[idx] = 0;
		index = idx;
		while (index < vecsize) {
			alt_vector[idx] += vector[index];
			index += stride;
		}
	

// CPU FUNCTIONS
float vector_reduction_CPU(const float * vector, const int n) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += vector[i];
  }
  return sum;
}


void sum_abs_rows_CPU(float * matrix, float * rowsum, int N, int M) {
    // The return value will be the matrix of rowsums
    for (int i = 0; i < N; ++i) {
        rowsum[i] = 0;
        for (int j = 0; j < M; ++j) {
            rowsum[i] += std::abs(matrix[i*N + j]);
        }
    }
}

void sum_abs_cols_CPU(float * matrix, float * colsum, int N, int M) {
    // The return value will be the matrix of rowsums
    for (int i = 0; i < M; ++i) {
        colsum[i] = 0;
        for (int j = 0; j < N; ++j) {
            colsum[i] += std::abs(matrix[j*M + i]);
        }
    }
}

// HELPER FUNCTIONS
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, long unsigned int & seed, struct timeval & start_time, int & print_time) {
  int c;
  unsigned int tmp;

  // Using getopt to parse the command line with options:
  // n - dimension of n
  // m - dimension of m
  // r - seed RNG with time(NULL)
  // h - help
  while ((c = getopt(argc, argv, "n:m:rth")) != -1) {
    switch(c) {
      case 'n':
        tmp = std::stoi(optarg); 
        if ((tmp > 1) && (tmp < MAX_DIM)) {
          n = tmp; 
        } else {
          std::cout << "Maximum dimension exceeded, using n = " << n << std::endl;
        }
        break;

      case 'm':
        tmp = std::stoi(optarg); 
        if ((tmp > 1) && (tmp < MAX_DIM)){
          m = tmp; 
        } else {
          std::cout << "Maximum dimension exceeded, using m = " << m << std::endl;
        }
        break;

        // Seed the RNG with microsecond time
      case 'r':
        gettimeofday(&start_time, NULL);
        seed = start_time.tv_usec;
        std::cout << "Seeding with value: " << seed << "\n" << std::endl;
        break;

      case 't':
        gettimeofday(&start_time, NULL);
        print_time = 1;
        break;
      
      case 'h':
        std::cout << "Usage: ./cpu_calc [-n ndim] [-m mdim] [-r (seed with time value?)] [-h (help)]" <<std::endl;
        exit(EXIT_FAILURE);

      case '?':
        std::cerr << "Unrecognized input!\n";
        exit(EXIT_FAILURE);
    }
  }
}

void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M) {
	if (N > 100 || M > 100) {
		return;
	}	

	for (int i = 0; i < N; i++) {
		std::cout << " | ";
		for (int j = 0; j < M; j++) 
				std::cout << std::setw(7) << std::setprecision(2)  << A[i*M + j];
		std::cout << " |\n";
	}
	std::cout << "\n";
}

