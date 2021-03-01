#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>
#include <string>

#define MAX_DIM 1000000

float * transpose_matrix(float *, float *, const int nrows, const int mcols);
void sum_abs_rows(float *, float *, const int, const int);
void sum_abs_cols(float *, float *, const int, const int);
void abs_value(float *, const int, const int);
float vector_reduction(const float *, const int);
void print_matrix(const float * matrix, const int n, const int m);
void parse_command_line(const int argc, char ** argv, unsigned int &, unsigned int &, long unsigned int &, struct timeval &, int &);

int main(int argc, char * argv[]) {

  // Default values for n, m
	unsigned int n {10}, m {10};
  
  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};
  struct timeval start_time;
	
  // Default seed
	long unsigned int seed {123456};
  
  // Get optional parameters
  parse_command_line(argc, argv, n, m, seed, start_time, print_time);

  std::cout << "\n\n";
	std::cout << "Allocating " << n << " x " << m << " matrix." << std::endl;
  
  float * matrix {new float[n*m]};

  // Seed RNG
	srand48(seed);

  // Populate matrix with values from [-10.0, 10.0]
  for (unsigned int i = 0; i < n*m; i++)
    matrix[i] = (float) (drand48()*20.0) - 10.0;


  std::cout << "\n\n";
  //print_matrix(matrix, n, m);
  float * transpose {new float[n*m]};
  transpose_matrix(matrix, transpose, n, m);
  std::cout << "\n\n";
  //print_matrix(transpose, m, n);
  float * row_sums {new float[m]};
  sum_abs_rows(matrix, row_sums, m, n);
  std::cout << "\n\nRow sums:\n\n";
  //print_matrix(row_sums, m, 1);
  float * col_sums {new float[n]};
  sum_abs_cols(matrix, col_sums, m, n);
  std::cout << "\n\nColumn Sums:\n\n";
  //print_matrix(col_sums, 1, n);
  std::cout << "\n\n";

  std::cout << "Sum of column sums: " << std::setprecision(8) << vector_reduction(col_sums, n) << std::endl;
  std::cout << "Sum of row sums: " << vector_reduction(row_sums, m) << std::endl;


	std::cout << "Deleting " << n << " x " << m << " matrix." << std::endl;
  delete[] matrix;
  delete[] transpose;
  delete[] col_sums;
  delete[] row_sums;
	return 0;
	
}

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
        if ((tmp > 1) && tmp < (MAX_DIM)){
          n = tmp; 
        } else {
          std::cout << "Maximum dimension exceeded, using n = 10" << std::endl;
        }
        break;

      case 'm':
        tmp = std::stoi(optarg); 
        if ((tmp > 1) && tmp < (MAX_DIM)){
          m = tmp; 
        } else {
          std::cout << "Maximum dimension exceeded, using m = 10" << std::endl;
        }
        break;

        // Seed the RNG with microsecond time
      case 'r':
        gettimeofday(&start_time, NULL);
        seed = start_time.tv_usec;
        std::cout << "Seeding with value: " << seed << std::endl;
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

void print_matrix(const float * matrix, const int n, const int m) {
  std::cout << std::setprecision(3);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << std::setw(8) << matrix[i*m + j];
    }
    std::cout << "\n";
  }
}

float vector_reduction(const float * vector, const int n) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += vector[i];
  }
  return sum;
}

void abs_value(float* matrix, const int n, const int m) {
  for (int i = 0; i < n*m; ++i) {
    matrix[i] = std::abs(matrix[i]);
  }
}

void sum_abs_rows(float * matrix, float * rowsum, const int nrows, const int mcols) {
  abs_value(matrix, nrows, mcols);
  for (int i = 0; i < nrows; ++i) {
    rowsum[i] = vector_reduction((matrix+(i*mcols)), mcols);
  }
}

void sum_abs_cols(float * matrix, float * colsum, const int nrows, const int mcols) {
  float * transpose {new float[nrows*mcols]};
  transpose_matrix(matrix, transpose, nrows, mcols);
  sum_abs_rows(matrix, colsum, mcols, nrows);
  delete[] transpose;
}

float * transpose_matrix(float * matrix, float * transposed_matrix, const int nrows, const int mcols) {
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < mcols; ++j) {
      transposed_matrix[nrows*j + i] = matrix[i*mcols + j];
    }
  }
  return transposed_matrix;
}
