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


void cpu_loop_dotp8(int K, double *device_x, double **device_y, int N, double *gpu_results){
	for(int i = 0; i < int(K/8); i++)
		gpu_dotp8_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, &device_y[8*i], N, &gpu_results[8*i]);	
}


void cpu_cublas_dotp(cublasHandle_t h, int K, int N, double* device_x, double** host_y, double* results){
	for (size_t i=0; i<K; ++i) {
		cublasDdot(h, N, device_x, 1, host_y[i], 1, results + i);
	}	
}

 
 
// __________________________ Execution wrapper ___________________________
// ________________________________________________________________________

template<typename CALLABLE, typename ...ARGS>
double execution_wrapper(int grid_size,
												 int block_size,
												 int repetitions,
												 bool device_function,
												 CALLABLE function, 
												 ARGS&& ... arg_pack)
{
	Timer single_timer;	
	double elapsed_time = 0;
	double median_time = 0;
	
	// vector of runtimes to calculate median time
	std::vector<double> runtimes;		
							
	for(int j=0; j<repetitions; j++){
		cudaDeviceSynchronize(); //make sure gpu is ready
		single_timer.reset();
		
		if (device_function)
			function<<<grid_size,block_size>>>(std::forward<ARGS>(arg_pack) ...);
		else
			function(std::forward<ARGS>(arg_pack) ...);
			
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
	bool check_results = false;
	
	const std::vector<size_t> N = {size_t(1e3), size_t(3e3), 
																size_t(1e4), size_t(3e4),
																size_t(1e5), size_t(3e5),
																size_t(1e6)};
													
	//const size_t N = 100000;
	//const size_t N = 10;
	const size_t K = 32;


	printf("N; K; time_cublas; time_dotp8\n");
	
	for(int ii = 0; ii < N.size(); ii++){
		printf("%zu; ",N[ii]);
		printf("%zu; ",K);

		//
		// Initialize CUBLAS:
		//
		//std::cout << "Init CUBLAS..." << std::endl;
		cublasHandle_t h;
		cublasCreate(&h);


		//
		// allocate host memory:
		//
		//std::cout << "Allocating host arrays..." << std::endl;
		double  *x = (double*)malloc(sizeof(double) * N[ii]);
		double **y = (double**)malloc(sizeof(double*) * K);
		for (size_t i=0; i<K; ++i) {
			y[i] = (double*)malloc(sizeof(double) * N[ii]);
		}
		double *results_ref  = (double*)malloc(sizeof(double) * K);
		double *results = (double*)malloc(sizeof(double) * K);


		//
		// allocate device memory
		//
		//std::cout << "Allocating device arrays..." << std::endl;
		double *device_x; cudaMalloc( (void **)(&device_x), sizeof(double)*N[ii]);
			
		// we create K pointers (to be used for device memory addresses) and store them in host memory
		double **host_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
		double **device_y; cudaMalloc(&device_y, sizeof(double*) * K);  // storing CUDA pointers on device!
		// we set our K pointers by using cudaMalloc 
		for (size_t i=0; i<K; ++i) {
			cudaMalloc( (void **)(&host_y[i]), sizeof(double)*N[ii]);
		}

		
		double *gpu_results;
		cudaMalloc(&gpu_results, K * sizeof(double));
			
		// fill host arrays with values
		cpu_init_vectors(x, y, N[ii], K );
			
		// Reference calculation on CPU
		if (check_results)
			cpu_dotp(x, y, results_ref, N[ii], K);
		
		//
		// Copy data to GPU
		//
		//std::cout << "Copying data to device..." << std::endl;
		cudaMemcpy(device_x, x, sizeof(double)*N[ii], cudaMemcpyHostToDevice);
		
		// copy the pointers to the device
		cudaMemcpy(device_y, host_y, K*sizeof(double*),cudaMemcpyHostToDevice);
		// copy the actual arrays to the device
		for (size_t i=0; i<K; ++i) {
			cudaMemcpy(host_y[i], y[i], sizeof(double)*N[ii], cudaMemcpyHostToDevice);
		}


		//
		// Let CUBLAS do the work:
		//
		for (size_t i=0; i<K; ++i)
			results[i] = 0;
		double time_cublas = 0;
		time_cublas = execution_wrapper(GRID_SIZE,BLOCK_SIZE,10,false, cpu_cublas_dotp, h, K, N[ii], device_x, host_y, results);
		printf("%5.8e; ", time_cublas);
		
		// kernel call for 1.2
		double time_dotp8 = 0;
		time_dotp8 = execution_wrapper(GRID_SIZE,BLOCK_SIZE,10,false,cpu_loop_dotp8,K,device_x, device_y, N[ii], gpu_results);
		printf("%5.8e\n", time_dotp8);

		

		//
		// Compare results
		//
		//std::cout << "Copying results back to host..." << std::endl;
		cudaMemcpy(results, gpu_results, sizeof(double)*K, cudaMemcpyDeviceToHost);

		if (check_results){
			for (size_t i=0; i<K; ++i) {
				std::cout << results_ref[i] << " on CPU, " << results[i] << " on GPU. Relative difference: " << fabs(results_ref[i] - results[i]) / results_ref[i] << std::endl;
			}		
		}
		
		//
		// Clean up:
		//
		//std::cout << "Cleaning up..." << std::endl;
		free(x);
		cudaFree(device_x);

		for (size_t i=0; i<K; ++i) {
			free(y[i]);
			cudaFree(host_y[i]);
		}
		free(y);
		free(host_y);
		cudaFree(device_y);

		free(results_ref);
		free(results);
		cudaFree(gpu_results);

	cublasDestroy(h);
		
	}
	
	cudaDeviceReset();  // for CUDA leak checker to work		
	return 0;
}
