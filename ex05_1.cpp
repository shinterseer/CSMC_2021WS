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


__global__
void gpu_set_to42(double *dotp){
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	
	//double thread_dotp[8] = {42};

	double* thread_dotp = new double[8];
	for(int i = 0; i < 8; i++) thread_dotp[i] = 42;
	
	if (thread_id_global == 0){		
		for(int i = 0; i < 8; i++) dotp[i] = thread_dotp[i];
		//dotp[0] = 42;
	}	
}


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
 
	
	// now the reduction inside the warp
	int shuffle_delta;
	int lane = threadIdx.x % warpSize;
	for(int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		if ((lane % (2*stride)) < stride)
			shuffle_delta = stride;
		else
			shuffle_delta = (-1)*stride;
		
		__syncwarp();
		thread_dotp += __shfl_down_sync(-1, thread_dotp, shuffle_delta);
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(dotp, thread_dotp);
	}	
}


__global__
//void gpu_dotp8_wshuffle(const double *x, double * const *y, const size_t size, double *dotp){
	// double * const * y ... I wanted const double **y, but for some reason, const has to be used like above	
void gpu_dotp8_wshuffle(const double *x, double *y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	
	// sum of entries
	//double thread_dotp[8] = {42};
	double* thread_dotp = new double[8];
	for(int i = 0; i < 8; i++) thread_dotp[i] = 0;
	//double dp0 = 0;
	//double dp1 = 0;
	//double dp2 = 0;
	//double dp3 = 0;
	//double dp4 = 0;
	//double dp5 = 0;
	//double dp6 = 0;
	//double dp7 = 0;
	double warp_exchanger;
	
	if (thread_id_global == 0){
		for(int i = 0; i < 8; i++) dotp[i] = 0;
	}
	
	
	for (unsigned int j = 0; j < 8; j++){
		for (unsigned int i = thread_id_global; i < size; i += thread_num){
			thread_dotp[j] += x[i] * y[j*size + i];
		}
	}
	
	
	// now the reduction inside the warp
	int shuffle_delta;
	int lane = threadIdx.x % warpSize;
	for(int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		if ((lane % (2*stride)) < stride)
			shuffle_delta = stride;
		else
			shuffle_delta = (-1)*stride;
			__syncwarp();
		
		for (unsigned int j = 0; j < 8; j++){
			//thread_dotp[j] += __shfl_down_sync(-1, thread_dotp[j], shuffle_delta);			
			warp_exchanger = thread_dotp[j];
			__syncwarp();
			thread_dotp[j] += __shfl_down_sync(-1, warp_exchanger, shuffle_delta);	
			//thread_dotp[j] += 1;
		}
	}
	
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		for (int j = 0; j < 8; j++){
			double write_this = thread_dotp[j];
			//atomicAdd(&dotp[j], write_this);
			atomicAdd(dotp+j, write_this);
		}
	}
	
	
	/*
	if (thread_id_global == 0){
		//for(int i = 0; i < 8; i++) dotp[i] = thread_dotp[i];
		dotp[0] = 42;
	}	
	*/
}



__global__
void gpu_dotp8_wshuffle2(const double *x, double *y, const size_t size, 
												double *dotp0,
												double *dotp1,
												double *dotp2,
												double *dotp3,
												double *dotp4,
												double *dotp5,
												double *dotp6,
												double *dotp7){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	
	// sum of entries
	//double thread_dotp[8] = {42};
	double* thread_dotp = new double[8];
	for(int i = 0; i < 8; i++) thread_dotp[i] = 0;
	//double dp0 = 0;
	//double dp1 = 0;
	//double dp2 = 0;
	//double dp3 = 0;
	//double dp4 = 0;
	//double dp5 = 0;
	//double dp6 = 0;
	//double dp7 = 0;
	double warp_exchanger;
	
	if (thread_id_global == 0){
		//for(int i = 0; i < 8; i++) dotp[i] = 0;
		*dotp0 = 0;
		*dotp1 = 0;
		*dotp2 = 0;
		*dotp3 = 0;
		*dotp4 = 0;
		*dotp5 = 0;
		*dotp6 = 0;
		*dotp7 = 0;
	}
	
	
	for (unsigned int j = 0; j < 8; j++){
		for (unsigned int i = thread_id_global; i < size; i += thread_num){
			thread_dotp[j] += x[i] * y[j*size + i];
		}
	}
	
	
	// now the reduction inside the warp
	int shuffle_delta;
	int lane = threadIdx.x % warpSize;
	for(int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		if ((lane % (2*stride)) < stride)
			shuffle_delta = stride;
		else
			shuffle_delta = (-1)*stride;
			__syncwarp();
		
		for (unsigned int j = 0; j < 8; j++){
			//thread_dotp[j] += __shfl_down_sync(-1, thread_dotp[j], shuffle_delta);			
			__syncwarp();
			warp_exchanger = thread_dotp[j];
			__syncwarp();
			thread_dotp[j] += __shfl_down_sync(-1, warp_exchanger, shuffle_delta);
			//thread_dotp[j] += 1;
		}
	}
	
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	//if ((threadIdx.x % warpSize) == 0){
	if ((thread_id_global % warpSize) == 0){
		atomicAdd(dotp0, thread_dotp[0]);
		atomicAdd(dotp1, thread_dotp[1]);
		atomicAdd(dotp2, thread_dotp[2]);
		atomicAdd(dotp3, thread_dotp[3]);
		atomicAdd(dotp4, thread_dotp[4]);
		atomicAdd(dotp5, thread_dotp[5]);
		atomicAdd(dotp6, thread_dotp[6]);
		atomicAdd(dotp7, thread_dotp[7]);
	}
	

	/*
	if (thread_id_global == 0){
		//for(int i = 0; i < 8; i++) dotp[i] = thread_dotp[i];
		dotp[0] = 42;
	}
	*/
	
	//*dotp7 = 3;
	//*dotp3 = 7777;
	
}



/*
void cpu_loop_call(double *cuda_x, double **cuda_y, double *gpu_results, int N, int K){
	// run my kernel
	int call_times = int(K/8);
	for(int i = 0; i < call_times; i++){
		gpu_dotp8_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(cuda_x, &cuda_y[8*i], N, &gpu_results[8*i]);	
	}
	
}
*/

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
	std::cout << "Allocating CUDA arrays..." << std::endl;
	double *cuda_x; cudaMalloc( (void **)(&cuda_x), sizeof(double)*N);
	//double **cuda_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
	//for (size_t i=0; i<K; ++i) {
	//	cudaMalloc( (void **)(&cuda_y[i]), sizeof(double)*N);
	//}
	double *gpu_results;
	double *gpu_results0, *gpu_results1, *gpu_results2, *gpu_results3, *gpu_results4, *gpu_results5, *gpu_results6, *gpu_results7;
	cudaMalloc(&gpu_results, K * sizeof(double));
	cudaMalloc(&gpu_results0, sizeof(double));
	cudaMalloc(&gpu_results1, sizeof(double));
	cudaMalloc(&gpu_results2, sizeof(double));
	cudaMalloc(&gpu_results3, sizeof(double));
	cudaMalloc(&gpu_results4, sizeof(double));
	cudaMalloc(&gpu_results5, sizeof(double));
	cudaMalloc(&gpu_results6, sizeof(double));
	cudaMalloc(&gpu_results7, sizeof(double));
	
	
	double *cuda_yy;  // avoiding double pointers
	cudaMalloc(&cuda_yy, N*K * sizeof(double));
	
	
	// fill host arrays with values
	cpu_init_vectors(x, y, N, K );
		
	// Reference calculation on CPU:
	cpu_dotp(x, y, results_ref, N, K);
	
	
	//
	// Copy data to GPU
	//
	std::cout << "Copying data to GPU..." << std::endl;
	cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
	for (size_t i=0; i<K; ++i) {
		//cudaMemcpy(cuda_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
		cudaMemcpy(&cuda_yy[i*N], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
	}

	//
	// Let CUBLAS do the work:
	//
	for (size_t i=0; i<K; ++i)
		results[i] = 0;
	
	/*
	std::cout << "Running dot products with CUBLAS..." << std::endl;
	for (size_t i=0; i<K; ++i) {
		cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results + i);
	}
	*/
	
	
	//cpu_loop_call(cuda_x, cuda_y, gpu_results, N, K);
	
	for( int i = 0; i < K; i++){
		//gpu_dotp_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(cuda_x, &(cuda_y[i][0]), N, &gpu_results[i]);
		//gpu_dotp_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(cuda_x, &cuda_yy[i*N], N, &gpu_results[i]);
	}
	
	cudaDeviceSynchronize();
	
	//gpu_dotp8_wshuffle<<<GRID_SIZE,BLOCK_SIZE>>>(cuda_x, cuda_yy, N, gpu_results);
	gpu_dotp8_wshuffle<<<1,128>>>(cuda_x, cuda_yy, N, gpu_results);
	/*gpu_dotp8_wshuffle2<<<1,64>>>(cuda_x, cuda_yy, N, 
																							gpu_results0,
																							gpu_results1,
																							gpu_results2,
																							gpu_results3,
																							gpu_results4,
																							gpu_results5,
																							gpu_results6,
																							gpu_results7);
	*/
	//gpu_set_to42<<<GRID_SIZE,BLOCK_SIZE>>>(gpu_results);
	//gpu_set_to42<<<1,1>>>(gpu_results);
	
	cudaMemcpy(results, gpu_results, K*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results, gpu_results0, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+1, gpu_results1, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+2, gpu_results2, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+3, gpu_results3, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+4, gpu_results4, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+5, gpu_results5, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+6, gpu_results6, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(results+7, gpu_results7, sizeof(double), cudaMemcpyDeviceToHost);

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
	cudaFree(cuda_x);

	for (size_t i=0; i<K; ++i) {
		free(y[i]);
		//cudaFree(cuda_y[i]);
	}
	cudaFree(cuda_yy);
	free(y);
	//free(cuda_y);

	free(results_ref);
	free(results);

	cublasDestroy(h);
	return 0;
}
