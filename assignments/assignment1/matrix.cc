#include <iostream> 
#include <iomanip> 
#include <stdlib.h> 
#include <sys/time.h> 

#include "matrix.h"

int main(int argc, char * argv[]) {

  // Default values for N, M
  unsigned int N {10000}, M {10000};
  
  // A boolean variable will tell us whether or not we want to print time
  int print_time {0};
  struct timeval start_time;
 
  // Default seed
  long unsigned int seed {123456};
  
  // Get optional parameters
  parse_command_line(argc, argv, N, M, seed, start_time, print_time);
  
  // Seed RNG
  srand48(seed);
  
  // Populate matrix with values from [-10.0, 10.0]
  float * A  {(float * ) malloc(sizeof(float)*N*M)};
  for (unsigned int i = 0; i < n*m; i++)
    A[i] = (float) drand48()*20.0 - 10.0;

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

