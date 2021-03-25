#ifndef MATRIX_H
#define MATRIX_H

// KERNELS
__global__ void sum_abs_rows_GPU(double * data, double * rowsum, const int N, const int M);
__global__ void sum_abs_cols_GPU(double * data, double * colsum, const int N, const int M);
__global__ void reduce0_GPU(double * vector_GPU, const int N);
__global__ void reduce1_GPU(double * vector_GPU, const int N);

double vector_reduction_GPU(double * vector_GPU, const int N, dim3 dimBlock, dim3 dimGrid);

// CPU MATRIX FUNCTIONS
double vector_reduction_CPU(const double * vector, const int n); 
void sum_abs_rows_CPU(double * matrix, double * rowsum, int N, int M);
void sum_abs_cols_CPU(double * matrix, double * colsum, int N, int M);

// HELPER FUNCTIONS
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, long unsigned int & seed, struct timeval & start_time, int & print_time, unsigned int & block_size);
void print_matrix_CPU(double * A, const unsigned int N, const unsigned int M);

#endif

