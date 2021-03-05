#ifndef MATRIX_H
#define MATRIX_H

// KERNELS
__global__ void sum_abs_rows_GPU(float * data, float * rowsum, int N, int M);
__global__ void sum_abs_cols_GPU(float * data, float * colsum, int N, int M);

// CPU MATRIX FUNCTIONS
float vector_reduction_CPU(const float * vector, const int n); 
void sum_abs_rows_CPU(float * matrix, float * rowsum, int N, int M);
void sum_abs_cols_CPU(float * matrix, float * colsum, int N, int M);

// HELPER FUNCTIONS
void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, long unsigned int & seed, struct timeval & start_time, int & print_time);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

#endif

