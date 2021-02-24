#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>

#define MAX_DIM 10000

int main(int argc, char * argv[]) {
	unsigned int n {10}, m {10};
	
	if (argc == 3) {
		n = std::stoi(argv[1]); ((n < 1) || n > (MAX_DIM)) ? (n = 10):0;
		m = std::stoi(argv[2]); ((m < 1) || m > (MAX_DIM)) ? (m = 10):0;
	}
	std::cout << "Using matrix dimension " << n << " x " << m << std::endl;

	unsigned int seed {123455};
	srand48(seed);

	std::cout << std::fixed;
	std::cout << std::setprecision(8);

	for (int i = 0; i < 10; i++) 
		std::cout << 20*drand48() - 10 << "\n";

	return 0;
	
}
