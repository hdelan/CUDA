#ifndef GPU_FUNCS_H_2323423
#define GPU_FUNCS_H_2323423

#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>


__global__ void GPU_exponentialIntegralDouble_3 (const double start, const double end, const int num_samples, const int start_n, const int max_n, double division, double * A);


void launch_on_one_card(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned block_size, const unsigned yblock, float & time_taken);

void launch_on_two_cards(double * & resultsDoubleGpu, const unsigned n, const unsigned numberOfSamples, const double a, const double b, const double division, const unsigned, const unsigned, float &);

#endif

