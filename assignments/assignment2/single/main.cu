#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

void cpu_rad_sweep(float * a, float * b, unsigned int n, unsigned int m);

void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, unsigned int & block_size);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

int main(int argc, char * argv[]) { 
  unsigned int n {32}, m {32}, block_size {32}, max_iters {100};
  long unsigned int seed {123};
  int print_time {0};

  parse_command_line(argc, argv, n, m, max_iters, seed, print_time, block_size);

  std::cout << "n: " << n << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "max iters: " << max_iters << "\n";
  std::cout << "print_time: " << print_time << "\n";
  std::cout << "block_size: " << block_size << "\n";

  float * A = (float *) calloc(n*m, sizeof(float));
  float * B = (float *) calloc(n*m, sizeof(float));

  // Set boundary conditions
  for (auto i=0;i<n;++i) {
    A[i*n] = B[i*n] = 1.0f*(float)(i+1)/(float)n;
    A[i*n+1] = B[i*n+1] = 0.80f*(float)(i+1)/(float)n;
  }

  print_matrix_CPU(A, n, m);
  for (int i=0;i<max_iters;++i){
    cpu_rad_sweep(B, A, n, m);
    cpu_rad_sweep(A, B, n, m);
  }

  std::cout << "\n\n";
  print_matrix_CPU(B, n, m);
  return 0;
}

void cpu_rad_sweep(float * a, float * b, unsigned int n, unsigned int m) {
  for (auto j=0;j<n;j++) {
    for (auto k=2;k<m-2;k++) {
      a[j*m+k] = (1.70f*b[j*m+k-2] + 1.40f*b[j*m+k-1] + b[j*m+k] + 0.60f*b[j*m+k+1] + 0.30*b[j*m+k+2])/5.0f;
    }
    a[j*m+m-2] = (1.40f*b[j*m+m-4] + 1.20f*b[j*m+m-3] + 0.8f*b[j*m+m-2] + 0.60f*b[j*m+m-1])/4.0f;
    a[j*m+m-1] = (1.40f*b[j*m+m-3] + b[j*m+m-2] + 0.6f*b[j*m+m-1])/3.0f;
  }
}
