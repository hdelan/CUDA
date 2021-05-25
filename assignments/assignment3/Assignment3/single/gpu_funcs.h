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


__global__ void GPU_exponentialIntegralFloat_3 (const float start, const float end, const int num_samples, const int start_n, const int max_n, float division, float * A);
__global__ void GPU_exponentialIntegralFloat_4_launch (const float start, const float end, const int num_samples, const int start_n, const int max_n, float division, float * A);
void GPU_exponentialIntegralFloat_4_execute (const float start, const float end, const int num_samples, const int n, const float division, const float psi_precomputed, float * A);


void launch_on_one_card(float * & resultsFloatGpu, const unsigned n, const unsigned numberOfSamples, const float a, const float b, const float division, const unsigned block_size, const unsigned yblock, float & time_taken);

void launch_on_two_cards(float * & resultsFloatGpu, const unsigned n, const unsigned numberOfSamples, const float a, const float b, const float division, const unsigned, const unsigned, float &);

#endif

