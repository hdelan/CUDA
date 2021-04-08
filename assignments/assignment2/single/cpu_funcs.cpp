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

unsigned int MAX_DIM = 20000;
unsigned int MAX_ITER = 2000;


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
 * \param:       print_time
 * \param:       block_size
 */
/* ----------------------------------------------------------------------------*/
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, unsigned int & block_size) {
  int c;
  unsigned int tmp;

  // Using getopt to parse the command line with options:
  // n - dimension of n
  // m - dimension of m
  // b - choose block size
  // r - seed RNG with time(NULL)
  // h - help
  while ((c = getopt(argc, argv, "n:m:b:p:rth")) != -1) {
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

      case 'p':
	tmp = std::stoi(optarg);
	if ((tmp > 1) && (tmp < MAX_ITER)) {
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
      
      case 'h':
        std::cout << "Usage: ./rad [-n ndim] [-m mdim] [-p num_iterations] [-r (seed with time value?)] [-h (help)]" <<std::endl;
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

	for (auto i = 0; i < N; i++) {
		std::cout << " | ";
		for (auto j = 0; j < M; j++) 
				std::cout << std::setw(7) << std::setprecision(2)  << A[i*M + j];
		std::cout << " |\n";
	}
	std::cout << "\n";
}

