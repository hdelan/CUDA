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
