#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>

#define MAX_DIM 1000000
#define BLOCK_SIZE 32

float vector_reduction(const float * vector, const int n);
void parse_command_line(const int argc, char ** argv, unsigned int &, unsigned int &, long unsigned int &, struct timeval &, int &);

class Matrix {
  public:
    Matrix() = delete;
    Matrix(const int n, const int m) : N {n}, M {m}, data {new float[n*m]}{};
    ~Matrix() { delete[] data;};
    Matrix(Matrix & m) : Matrix(m.N, m.M){
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          data[i*M + j] = m.data[i*M + j];
        }
      }
    };

    void print_matrix(){
      std::cout << "\n" << N << " x " << M << " matrix: \n\n";
      std::cout << std::setprecision(3);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          std::cout << std::setw(8) << data[i*M + j];
        }
        std::cout << "\n";
      }
    };

    float & operator[](int i){
      return data[i];
    }

    Matrix sum_abs_rows() {
      // The return value will be the matrix of rowsums
      Matrix rowsum(N, 1);
      for (int i = 0; i < N; ++i) {
        rowsum.data[i] = 0;
        for (int j = 0; j < M; ++j) {
          rowsum[i] += std::abs(data[i*N + j]);
        }
      }
      return rowsum;
    };

    Matrix sum_abs_cols() {
      Matrix colsum(M, 1);
      for (int i = 0; i < M; ++i) {
        colsum[i] = 0;
        for (int j = 0; j < M; ++j) {
          colsum[i] += std::abs(data[j*N + i]);
        }
      }
      return colsum;
    
    };

    float sum_abs_matrix() {
      float sum = 0.0;
      for (int i = 0; i < N*M; ++i) {
        sum += data[i];
      }
      return sum;
    };
    
  

    void send_data_to_GPU() {
	cudaMalloc((void **) &data_GPU, sizeof(float)*N*M);
    };
    
    void send_data_to_GPU() {
	cudaMemcpy(data_GPU, data, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    };

  protected:
    int N, M;
    float * data;
};

class Matrix_GPU : public Matrix {
public:
    Matrix_GPU() = delete;
    Matrix_GPU(const int n, const int m) : Matrix(n, m), block_size{BLOCK_SIZE} {
	    cudaMalloc((void **) &data_GPU, sizeof(float)*n*m);
	    dim3 dimBlock(block_size);
	    dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
    };
    Matrix_GPU(const int n, const int m, int block_size) : Matrix(n, m), block_size {block_size} {
	    cudaMalloc((void **) &data_GPU, sizeof(float)*n*m);
	    dim3 dimBlock(block_size);
	    dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
    	    };
    ~Matrix() { free(data_GPU);};

    void data_to_GPU() {
	cudaMemcpy(data_GPU, data, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    }

    void data_from_GPU() {
	cudaMemcpy(data, data_GPU, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
    }
    
    Matrix_GPU sum_abs_rows_GPU() {
      // The return value will be the matrix of rowsums
      Matrix_GPU rowsum(N, 1);
      for (int i = 0; i < N; ++i) {
        rowsum.data[i] = 0;
        for (int j = 0; j < M; ++j) {
          rowsum[i] += std::abs(data[i*N + j]);
        }
      }
      return rowsum;
    };

    Matrix_GPU sum_abs_cols_GPU() {
      Matrix colsum(M, 1);
      for (int i = 0; i < M; ++i) {
        colsum[i] = 0;
        for (int j = 0; j < M; ++j) {
          colsum[i] += std::abs(data[j*N + i]);
        }
      }
      return colsum;
    
    };



private:
    float * dataGPU;
    int block_size;
};

#endif
