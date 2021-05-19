#ifndef GPU_FUNCS_728349
#define GPU_FUNCS_728349

__global__ void gpu_rad_sweep5(float*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_rad_sweep6(float*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_get_averages(float * a, unsigned int n, unsigned int m, float * avg);

#endif
