#ifndef GPU_FUNCS_728349
#define GPU_FUNCS_728349

__global__ void gpu_rad_sweep5(double*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_rad_sweep6(double*, unsigned int, unsigned int, unsigned int);
__global__ void gpu_get_averages(double * a, unsigned int n, unsigned int m, double * avg);

#endif
