#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "timer.hpp"
#include <vector>
#include <algorithm>
  
#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define ALL_MASK 0xffffffff

//#define DEBUG

// __________________________ GPU Kernels _________________________________
// ________________________________________________________________________


/*
__global__
void gpu_dotp_wshuffle(const double *x, const double *y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	// sum of entries
	double thread_dotp = 0;
	
	if (thread_id_global == 0){
		*dotp = 0;
	}
 
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_dotp += x[i] * y[i];
	}
 	
	for(unsigned int stride = warpSize/2; stride > 0; stride /= 2){		
		__syncwarp();
		thread_dotp += __shfl_down_sync(ALL_MASK, thread_dotp, stride);
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(dotp, thread_dotp);
	}	
}
*/

/*
__global__
//void gpu_dotp8_wshuffle_old(const double *x, double * const *y, const size_t size, double *dotp){
	// double * const * y ... I wanted const double **y, but for some reason, const has to be used like above	
void gpu_dotp8_wshuffle_old(const double *x, double *y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
		double thread_dotp[8] = {0};
	for(int i = 0; i < 8; i++) thread_dotp[i] = 0;
	
	if (thread_id_global == 0){
		for(int i = 0; i < 8; i++) dotp[i] = 0;
	}
	
	
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		for (unsigned int j = 0; j < 8; j++){
			thread_dotp[j] += x[i] * y[j*size + i];
		}
	}
		
	for(int stride = warpSize/2; stride > 0; stride /= 2){		
		__syncwarp();
		for (unsigned int j = 0; j < 8; j++){
			thread_dotp[j] += __shfl_down_sync(ALL_MASK, thread_dotp[j], stride);			
		}
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		for (int j = 0; j < 8; j++){
			atomicAdd(&dotp[j], thread_dotp[j]);
		}
	}	
}
*/

__global__
void gpu_dotp8_wshuffle(const double *x, double * const *y, const size_t size, double *dotp){
	// double * const * y ... I wanted const double **y, but for some reason, const has to be used like above	
//void gpu_dotp8_wshuffle2(const double *x, double **y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	double thread_dotp[8] = {0};
	
	// initialize local dot products
	for(int i = 0; i < 8; i++) thread_dotp[i] = 0;
	
	// initialize result vector for atomicAdd()
	if (thread_id_global == 0){
		for(int i = 0; i < 8; i++) dotp[i] = 0;
	}
	
	// dot product
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		for (unsigned int j = 0; j < 8; j++){
			thread_dotp[j] += x[i] * y[j][i];
		}
	}
	
	// sumation stage1 (warp reduction)
	for(int stride = warpSize/2; stride > 0; stride /= 2){
		__syncwarp();
		for (unsigned int j = 0; j < 8; j++){
			thread_dotp[j] += __shfl_down_sync(ALL_MASK, thread_dotp[j], stride);			
		}
	}
	__syncwarp();
	
	// sumation stage2 (atomicAdd)	
	if ((threadIdx.x % warpSize) == 0){
		for (int j = 0; j < 8; j++){
			atomicAdd(&dotp[j], thread_dotp[j]);
		}
	}	
}

// __________________________ Toolbox _____________________________________
// ________________________________________________________________________


void cpu_dotp(double *x, double **y, double *results, size_t N, size_t K){
	//
	// Reference calculation on CPU:
	//
	for (size_t i=0; i<K; ++i) {
		results[i] = 0;
		for (size_t j=0; j<N; ++j) {
			results[i] += x[j] * y[i][j];
		}
	}    
}


void cpu_init_vectors(double *x, double **y, size_t N, size_t K ){
	//
	// fill host arrays with values
	//
	for (size_t j=0; j<N; ++j) {
		x[j] = 1 + j%K;
	}
	for (size_t i=0; i<K; ++i) {
		for (size_t j=0; j<N; ++j) {
			y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
		}
	}	
}

 
 
// __________________________ Execution wrapper ___________________________
// ________________________________________________________________________

template<typename KERNEL, typename ...ARGS>
double execution_wrapper(int grid_size,
												 int block_size,
												 int repetitions,
												 KERNEL gpu_kernel, 
												 ARGS... pass_this_to_kernel)
{
	Timer single_timer;	
	double elapsed_time = 0;
	double median_time = 0;
	
	// vector of runtimes to calculate median time
	std::vector<double> runtimes;		
							
	for(int j=0; j<repetitions; j++){
		cudaDeviceSynchronize(); //make sure gpu is ready
		single_timer.reset();
		
		gpu_kernel<<<grid_size,block_size>>>(pass_this_to_kernel...);

		cudaDeviceSynchronize(); //make sure gpu is done			
		elapsed_time = single_timer.get();
		runtimes.push_back(elapsed_time);
	}
	
	std::sort(runtimes.begin(), runtimes.end());
	median_time = runtimes[int(repetitions/2)];
	return median_time;
}



int main(void)
{
	//bool test_results = false;
	
	const size_t N = 100000;
	//const size_t N = 10;
	const size_t K = 8;

	//
	// Initialize CUBLAS:
	//
	std::cout << "Init CUBLAS..." << std::endl;
	cublasHandle_t h;
	cublasCreate(&h);


	//
	// allocate host memory:
	//
	std::cout << "Allocating host arrays..." << std::endl;
	double  *x = (double*)malloc(sizeof(double) * N);
	double **y = (double**)malloc(sizeof(double*) * K);
	for (size_t i=0; i<K; ++i) {
		y[i] = (double*)malloc(sizeof(double) * N);
	}
	double *results_ref  = (double*)malloc(sizeof(double) * K);
	double *results = (double*)malloc(sizeof(double) * K);


	//
	// allocate device memory
	//
	std::cout << "Allocating device arrays..." << std::endl;
	double *device_x; cudaMalloc( (void **)(&device_x), sizeof(double)*N);
		
	// we create K pointers (to be used for device memory addresses) and store them in host memory
	//double **cuda_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
	double **host_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
	double **device_y; cudaMalloc(&device_y, sizeof(double*) * K);  // storing CUDA pointers on device!
	// we set our K pointers by using cudaMalloc 
	for (size_t i=0; i<K; ++i) {
		cudaMalloc( (void **)(&host_y[i]), sizeof(double)*N);
	}

	
	double *gpu_results;
	cudaMalloc(&gpu_results, K * sizeof(double));
		
	// fill host arrays with values
	cpu_init_vectors(x, y, N, K );
		
	// Reference calculation on CPU:
	cpu_dotp(x, y, results_ref, N, K);
	
	
	//
	// Copy data to GPU
	//
	std::cout << "Copying data to device..." << std::endl;
	cudaMemcpy(device_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
	
	// copy the pointers to the device
	cudaMemcpy(device_y, host_y, K*sizeof(double*),cudaMemcpyHostToDevice);
	// copy the actual arrays to the device
	for (size_t i=0; i<K; ++i) {
		cudaMemcpy(host_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
	}

	//
	// Let CUBLAS do the work:
	//
	for (size_t i=0; i<K; ++i)
		results[i] = 0;
	
	/*
	std::cout << "Running dot products with CUBLAS..." << std::endl;
	for (size_t i=0; i<K; ++i) {
		cublasDdot(h, N, device_x, 1, host_y[i], 1, results + i);
	}
	*/
		
	//cpu_loop_call(cuda_x, cuda_y, gpu_results, N, K);
	
	cudaDeviceSynchronize();
	
	gpu_dotp8_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, N, gpu_results);
	
	cudaMemcpy(results, gpu_results, K*sizeof(double), cudaMemcpyDeviceToHost);
	
	//
	// Compare results
	//
	std::cout << "Copying results back to host..." << std::endl;
	for (size_t i=0; i<K; ++i) {
		std::cout << results_ref[i] << " on CPU, " << results[i] << " on GPU. Relative difference: " << fabs(results_ref[i] - results[i]) / results_ref[i] << std::endl;
	}

	
	//
	// Clean up:
	//
	std::cout << "Cleaning up..." << std::endl;
	free(x);
	cudaFree(device_x);

	for (size_t i=0; i<K; ++i) {
		free(y[i]);
		cudaFree(host_y[i]);
	}
	free(host_y);
	cudaFree(device_y);
	free(host_y);	
	free(y);

	free(results_ref);
	free(results);

	cublasDestroy(h);
	cudaDeviceReset();  // for CUDA leak checker to work		
	return 0;
}
