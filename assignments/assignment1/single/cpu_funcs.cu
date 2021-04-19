/**
 * \file:        cpu_funcs.cu
 * \brief:       CUDA Assignment 1:
 *               Some CPU functions for summing rows, summing columns and performing vector reductions, as well as helpers.
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-03-25
 */

#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <unistd.h> 
#include <sys/time.h> 

#include "matrix.h"

unsigned int MAX_DIM = 100000;

// CPU FUNCTIONS


/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to perform a vector reduction in serial on CPU
 *
 * \param:       vector
 * \param:       n
 *
 * \returns      Sum
 */
/* ----------------------------------------------------------------------------*/
float vector_reduction_CPU(const float * vector, const int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += vector[i];
  }
  return sum;
}


/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to compute the rowsums in serial on CPU
 *
 * \param:       matrix         The matrix to be rowsummed
 * \param:       rowsum         The N-length returned rowsum -- "a column vector"
 * \param:       N              The number of rows
 * \param:       M              The number of columns
 */
/* ----------------------------------------------------------------------------*/
void sum_abs_rows_CPU(float * matrix, float * rowsum, int N, int M) {
    // The return value will be the matrix of rowsums
    for (int i = 0; i < N; ++i) {
        rowsum[i] = 0.0f;
        for (int j = 0; j < M; ++j) {
            rowsum[i] += std::fabs(matrix[i*M + j]);
        }
    }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to compute column sums in serial on CPU
 *
 * \param:       matrix
 * \param:       colsum        The M-length colsum to be returned
 * \param:       N
 * \param:       M
 */
/* ----------------------------------------------------------------------------*/
void sum_abs_cols_CPU(float * matrix, float * colsum, int N, int M) {
    // The return value will be the matrix of rowsums
    for (int i = 0; i < M; ++i) {
        colsum[i] = 0.0f;
        for (int j = 0; j < N; ++j) {
            colsum[i] += std::fabs(matrix[j*M + i]);
        }
    }
}

// HELPER FUNCTIONS


/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to parse the command line for optional parameters
 *
 * \param:       argc
 * \param:       argv
 * \param:       n
 * \param:       m
 * \param:       seed
 * \param:       start_time
 * \param:       print_time
 * \param:       block_size
 */
/* ----------------------------------------------------------------------------*/
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, long unsigned int & seed, struct timeval & start_time, int & print_time, unsigned int & block_size) {
  int c;
  unsigned int tmp;

  // Using getopt to parse the command line with options:
  // n - dimension of n
  // m - dimension of m
  // b - choose block size
  // r - seed RNG with time(NULL)
  // h - help
  while ((c = getopt(argc, argv, "n:m:b:rth")) != -1) {
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
	//if (N > 100 || M > 100) {
	//	return;
	//}	

	for (int i = 0; i < N; i++) {
		std::cout << " | ";
		for (int j = 0; j < M; j++) 
				std::cout << std::setw(7) << std::setprecision(2)  << A[i*M + j];
		std::cout << " |\n";
	}
	std::cout << "\n";
}

