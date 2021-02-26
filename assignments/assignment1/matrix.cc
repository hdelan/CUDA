#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>
#include <string>

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

    void set_data_value(float val, int idx) {
      if (idx >= nrows * mcols || idx < 0) {
        std::cerr << "Trying to access data beyond bounds\n";
        return;
      }
      data[idx] = val;
    };

    Matrix transpose() {
      Matrix trans(mcols, nrows);
      for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < mcols; ++j) {
          trans.set_data_value(data[i*mcols + j], j*nrows + i);
        }
      }
      return trans;
    };

    void abs_value() {
      for (int i = 0; i < nrows*mcols; ++i) {
        set_data_value(std::abs(data[i]), i);
      }
    };

    Matrix sum_abs_rows() {
      Matrix rowsum(nrows, 1);
      abs_value();
      for (int i = 0; i < nrows; ++i) {
        rowsum.set_data_value(vector_reduction((data+(i*mcols)), mcols), i);
      }
      return rowsum;
    };

    Matrix sum_abs_cols() {
      Matrix trans {transpose()};
      return (trans.sum_abs_rows()).transpose();
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
    A.set_data_value((float) (drand48()*20.0) - 10.0, i);
	

  A.print_matrix();
  A.transpose().print_matrix();

  Matrix colsum = A.sum_abs_cols();
  Matrix rowsum = A.sum_abs_rows();

  colsum.print_matrix();
  std::cout << "\n\nSum of colsums: " << colsum.sum_abs_matrix() << "\n\n";
  rowsum.print_matrix();
  std::cout << "\n\nSum of rowsums: " << rowsum.sum_abs_matrix() << "\n\n";

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

