/**
 * \file:        cpu_funcs.cu
 * \brief:       CUDA Assignment 1:
 *               Some CPU functions for summing rows, summing columns and performing vector reductions, as well as helpers.
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-03-25
 */

#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

unsigned int MAX_DIM = 20000;
unsigned int MAX_ITER = 2000;

// RADIATOR FUNCTIONS

template <typename T>
void cpu_rad_sweep1(T * a, unsigned int n, unsigned int m, unsigned int iters, T * b) {
  T * tmp;
  for (auto i=0u;i<iters;i++) {
    for (auto j=0u;j<n;j++) {
      for (auto k=2u;k<m-2;k++) {
        b[j*m+k] = (1.70f*a[j*m+k-2] + 1.40f*a[j*m+k-1] + a[j*m+k] + 0.60f*a[j*m+k+1] + 0.30*a[j*m+k+2])/5.0f;
      }
      // Getting the end terms
      b[j*m+m-2] = (1.70f*a[j*m+m-4] + 1.40f*a[j*m+m-3] + a[j*m+m-2] + 0.60f*a[j*m+m-1]+0.30f*a[j*m])/5.0f;
      b[j*m+m-1] = (1.70f*a[j*m+m-3] + 1.40f*a[j*m+m-2] + a[j*m+m-1] + 0.60f*a[j*m]+0.30f*a[j*m+1])/5.0f;
    }
    tmp = a;
    a = b;
    b = tmp;
  }

  // Make both matrices identical on exit
  for (auto j=0u;j<n;j++) {
    for (auto k=2u;k<m;k++) {
      // a has just been operated on so we store the same values in b
      b[j*m+k] = a[j*m+k];
    }
  }
  

}

void get_averages(float * a, unsigned int n, unsigned int m, float * avg) {
  for (auto i=0u;i<n;i++) {
    avg[i] = 0.0f;
    for (auto j=0u;j<m;j++) {
      avg[i] += a[i*m+j];
    }
    avg[i] /= (float) m;
  }
}

// HELPER FUNCTIONS


void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, int & cpu_calc, unsigned int & block_size, int & write_file) {
  int c;
  unsigned int tmp;

  // Using getopt to parse the command line with options:
  // n - dimension of n
  // m - dimension of m
  // b - choose block size
  // r - seed RNG with time(NULL)
  // h - help
  while ((c = getopt(argc, argv, "n:m:b:p:rwcth")) != -1) {
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

        // Choose the blocksize?
      case 'b':
        tmp = std::stoi(optarg);
        if ((tmp > 1) && (tmp < 1025)) {
          block_size = tmp;
        } else {
          std::cout << "Invalid block size, using default " << block_size << std::endl;
        }
        break;

        // Number of iterations
      case 'p':
        tmp = std::stoi(optarg);
        if ((tmp >= 1) && (tmp <= MAX_ITER)) {
          iters = tmp;
        } else {
          std::cout << "Invalid num_iters, using default " << iters << std::endl;
        }
        break;

        // Seed the RNG with microsecond time
      case 'r':
        struct timeval start_time;
        gettimeofday(&start_time, NULL);
        seed = start_time.tv_usec;
        std::cout << "Seeding with value: " << seed << "\n" << std::endl;
        break;

      case 't':
        print_time = 1;
        break;

      case 'c': 
        cpu_calc = 0;
        break; 
      
      case 'w': 
        write_file = 1;
        break; 

      case 'h':
        std::cout << "Usage: ./rad [-n ndim] [-m mdim] [-p num_iterations] [-c (skip cpu calculation?)] [-r (seed with time value?)] [-h (help)]" <<std::endl;
        exit(EXIT_FAILURE);

      case '?':
        std::cerr << "Unrecognized input!\n";
        exit(EXIT_FAILURE);
    }
  }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to print a matrix if it is smaller than 100 x 100
 *
 * \param:       A
 * \param:       N
 * \param:       M
 */
/* ----------------------------------------------------------------------------*/
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M) {
  if (N > 100 || M > 100) {
    return;
  }	

  for (auto i = 0u; i < N; i++) {
    std::cout << " | ";
    for (auto j = 0u; j < M; j++) 
      std::cout << std::setw(7) << std::setprecision(2)  << A[i*M + j];
    std::cout << " |\n";
  }
  std::cout << "\n";
}

void print_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M) {
  std::ofstream f1;
  f1.open(filename);
  f1 << N << " " << M << '\n';
  for (auto i=0u;i<N;i++) {
    for (auto j=0u;j<M;j++) {
      f1 <<  " " << A[i*M+j] << " ";
    }
    f1 << '\n';
  }
  f1.close();
}

void read_matrix_from_file(std::string filename, float * A) {
  // A needs to be allocated already 
  unsigned int N, M;

  std::ifstream f1;
  f1.open(filename);
  if (f1.fail()) throw std::exception();
  f1 >> N >> M;

  for (auto i=0u;i<N;i++) {
    for (auto j=0u;j<M;j++) {
      f1 >> A[i*M+j];
    }
  }
  f1.close();
}

void print_sparse_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M, const unsigned int iters) {
  if (2+4*iters >= M) {
    print_matrix_to_file(filename, A, N, M);
    return;
  }
  std::ofstream f1;
  f1.open(filename);
  f1 << N << " " << M << " " << iters << '\n';
  for (auto i=0u;i<N;i++) {
    // Writing left propagation
    for (auto j=0u;j<2*iters+2;j++) {
      f1 <<  " " << A[i*M+j] << " ";
    }
    
    // Writing right propagation
    for (auto j=M-2*iters;j<M;j++) {
      f1 <<  " " << A[i*M+j] << " ";
    }

    f1 << '\n';
  }
  f1.close();
}

void read_sparse_matrix_from_file(std::string filename, float * A) {
  // A needs to be allocated already with its elements initialized to zero
  unsigned int N, M, iters;

  std::ifstream f1;
  f1.open(filename);
  if (f1.fail()) throw std::exception();
  
  f1 >> N >> M >> iters;
  if (2+4*iters >= M) {
    read_matrix_from_file(filename, A);
    return;
  }

  for (auto i=0u;i<N;i++) {
    // Reading left propagation
    for (auto j=0u;j<2*iters+2;j++) {
      f1 >> A[i*M+j];
    }
    
    // Reading right propagation
    for (auto j=M-2*iters;j<M;j++) {
      f1 >> A[i*M+j];
    }
  }
  f1.close();
}

//template void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);
//template void print_matrix_CPU(double * A, const unsigned int N, const unsigned int M);
template void cpu_rad_sweep1(float * a, unsigned int n, unsigned int m, unsigned int iters, float * b);
template void cpu_rad_sweep1(double * a, unsigned int n, unsigned int m, unsigned int iters, double * b);
