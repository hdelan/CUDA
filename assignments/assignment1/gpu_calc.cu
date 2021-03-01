#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"

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
  
  // Seed RNG
	srand48(seed);
  
  // Populate matrix with values from [-10.0, 10.0]
  Matrix A(n, m);
  for (unsigned int i = 0; i < n*m; i++)
    A[i] = (float) drand48()*20.0 - 10.0;
	

  //A.print_matrix();
  //A.transpose().print_matrix();

  Matrix colsum = A.sum_abs_cols();
  Matrix rowsum = A.sum_abs_rows();

  //colsum.print_matrix();
  std::cout << "\n\nSum of colsums: " << colsum.sum_abs_matrix() << "\n\n";
  rowsum.print_matrix();
  std::cout << "\n\nSum of rowsums: " << rowsum.sum_abs_matrix() << "\n\n";
  
  if (print_time == 1) {
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    std::cout << "\n\nTime taken: " << std::setprecision(8) << ((double) end_time.tv_sec - start_time.tv_sec) + (((double)end_time.tv_usec - start_time.tv_usec)/1000000.0) << " seconds\n\n";
  }
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

float vector_reduction(const float * vector, const int n) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += vector[i];
  }
  return sum;
}

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>

#define MAX_DIM 1000000

float vector_reduction(const float * vector, const int n);
void parse_command_line(const int argc, char ** argv, unsigned int &, unsigned int &, long unsigned int &, struct timeval &, int &);

class Matrix {
  public:
    Matrix() = delete;
    Matrix(const int n, const int m) : nrows {n}, mcols {m}, data {new float[n*m]}{};
    ~Matrix() { delete[] data; };
    Matrix(Matrix & m) : Matrix(m.nrows, m.mcols){
      for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < mcols; ++j) {
          data[i*mcols + j] = m.data[i*mcols + j];
        }
      }
    };

    void print_matrix(){
      std::cout << "\n" << nrows << " x " << mcols << " matrix: \n\n";
      std::cout << std::setprecision(3);
      for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < mcols; ++j) {
          std::cout << std::setw(8) << data[i*mcols + j];
        }
        std::cout << "\n";
      }
    };

    float & operator[](int i){
      return data[i];
    }

    Matrix sum_abs_rows() {
      // The return value will be the matrix of rowsums
      Matrix rowsum(nrows, 1);
      for (int i = 0; i < nrows; ++i) {
        rowsum.data[i] = 0;
        for (int j = 0; j < mcols; ++j) {
          rowsum[i] += std::abs(data[i*nrows + j]);
        }
      }
      return rowsum;
    };

    Matrix sum_abs_cols() {
      Matrix colsum(mcols, 1);
      for (int i = 0; i < mcols; ++i) {
        colsum[i] = 0;
        for (int j = 0; j < mcols; ++j) {
          colsum[i] += std::abs(data[j*nrows + i]);
        }
      }
      return colsum;
    
    };

    float sum_abs_matrix() {
      float sum = 0.0;
      for (int i = 0; i < nrows*mcols; ++i) {
        sum += data[i];
      }
      return sum;
    };

  private:
    int nrows, mcols;
    float * data;
};

#endif
