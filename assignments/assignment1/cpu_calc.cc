#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <chrono>
#include <string>

#define MAX_DIM 10000

int main(int argc, char * argv[]) {
	unsigned int n {10}, m {10}, tmp;
	
	long unsigned int seed {123456};
  
  int c;

  // Using getopt to parse the command line with options:
  // n - dimension of n
  // m - dimension of m
  // r - seed RNG with time(NULL)
  // h - help
  while ((c = getopt(argc, argv, "n:m:rh")) != -1) {
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

      case 'r':
        seed = time(NULL);
        std::cout << "Seeding with value: " << seed << std::endl;
        break;

      case 'h':
        std::cout << "Usage: ./cpu_calc [-n ndim] [-m mdim] [-r (seed with time value?)] [-h (help)]" <<std::endl;
        exit(EXIT_FAILURE);

      case '?':
        std::cerr << "Unrecognized input!\n";
        exit(EXIT_FAILURE);
    }
  }

	std::cout << "Using matrix dimension " << n << " x " << m << std::endl;

	srand48(seed);

	std::cout << std::fixed;
	std::cout << std::setprecision(8);

	for (int i = 0; i < n; i++) 
		std::cout << 20*drand48() - 10 << "\n";

	return 0;
	
}
