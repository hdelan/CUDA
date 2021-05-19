#ifndef CPU_FUNCS_812830
#define CPU_FUNCS_812830

void print_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M);
void read_matrix_from_file(std::string filename, float * A);

void print_sparse_matrix_to_file(std::string filename, float * A, const unsigned int N, const unsigned int M, const unsigned int iters);
void read_sparse_matrix_from_file(std::string filename, float * A);

template <typename T>
void cpu_rad_sweep1(T*, unsigned int, unsigned int, unsigned int, T*);
void get_averages(float * a, unsigned int n, unsigned int m, float * avg);

void diff_matrices(float *A, float *B, unsigned int n, unsigned int m);

void parse_command_line(const int argc, char ** argv, unsigned int & n, unsigned int & m, unsigned int & iters, long unsigned int & seed, int & print_time, int & cpu_calc, unsigned int & block_size, int & write_file, int & avg);
void print_matrix_CPU(float * A, const unsigned int N, const unsigned int M);

#endif
