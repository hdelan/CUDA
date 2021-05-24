#ifndef GPU_FUNCS_H_2323423
#define GPU_FUNCS_H_2323423

//extern __constant__ double C_eulerConstant;

__global__ void GPU_exponentialIntegralDouble_1 (const double start, const double end, const int num_samples, double division, double * A);
__global__ void GPU_exponentialIntegralDouble_2 (const double start, const double end, const int num_samples, const int max_n, double division, double * A);
__global__ void GPU_exponentialIntegralDouble_3 (const double start, const double end, const int num_samples, const int start_n, const int max_n, double division, double * A);
__global__ void GPU_exponentialIntegralDouble_4 (const double start, const double end, const int num_samples, const int max_n, double division, double * A);

#endif
