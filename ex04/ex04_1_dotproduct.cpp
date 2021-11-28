#include <stdio.h>
#include <iostream>
#include "timer.hpp"
#include <vector>
#include <algorithm>
#include <string>

#define BLOCK_SIZE 256


// ########################################################################
// ################## The Kernels #########################################
// ########################################################################


__global__
void gpu_dotproduct_atomicAdd(const double *gpu_x, const double *gpu_y, const size_t size, double *gpu_result){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread_id_global == 0)
		*gpu_result = 0;
	
	__shared__ double shared_m[BLOCK_SIZE];
		
	// I think, this is the right way:
	double thread_dp = 0;
	for (unsigned int i = thread_id_global; i<size; i += blockDim.x * gridDim.x)
		thread_dp += gpu_x[i] * gpu_y[i];
	shared_m[threadIdx.x] = thread_dp;
		
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared_m[threadIdx.x] += shared_m[threadIdx.x + stride];
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_result, shared_m[0]);		
		//gpu_result_stage1[blockIdx.x] = shared_m[0];
	}	
}


__global__
void gpu_stuff_sharedm(const double *gpu_x, const size_t size, 
										  double *gpu_sum, double *gpu_abssum, double *gpu_squaresum, double *gpu_numzeros){
	
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	__shared__ double shared_sum[BLOCK_SIZE];
	__shared__ double shared_abssum[BLOCK_SIZE];
	__shared__ double shared_squaresum[BLOCK_SIZE];
	__shared__ double shared_numzeros[BLOCK_SIZE];
		
	// sum of entries
	double thread_sum = 0;
	double thread_abssum = 0;
	double thread_squaresum = 0;
	double thread_numzeros = 0;
	
	if (thread_id_global == 0){
		*gpu_sum = 0;
		*gpu_abssum = 0;
		*gpu_squaresum = 0;
		*gpu_numzeros = 0;
	}
		
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_sum += gpu_x[i];
		thread_abssum += std::abs(gpu_x[i]);
		thread_squaresum += gpu_x[i] * gpu_x[i];
		if (gpu_x[i] == 0)
			thread_numzeros += 1.;	
	}

	shared_sum[threadIdx.x] = thread_sum;
	shared_abssum[threadIdx.x] = thread_abssum;
	shared_squaresum[threadIdx.x] = thread_squaresum;
	shared_numzeros[threadIdx.x] = thread_numzeros;
		
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
			shared_abssum[threadIdx.x] += shared_abssum[threadIdx.x + stride];
			shared_squaresum[threadIdx.x] += shared_squaresum[threadIdx.x + stride];
			shared_numzeros[threadIdx.x] += shared_numzeros[threadIdx.x + stride];			
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_sum, shared_sum[0]);
		atomicAdd(gpu_abssum, shared_abssum[0]);
		atomicAdd(gpu_squaresum, shared_squaresum[0]);
		atomicAdd(gpu_numzeros, shared_numzeros[0]);
	}	
}


__global__
void gpu_stuff_wshuffle(const double *gpu_x, const size_t size, 
										  double *gpu_sum, double *gpu_abssum, double *gpu_squaresum, double *gpu_numzeros){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	// sum of entries
	double thread_sum = 0;
	double thread_abssum = 0;
	double thread_squaresum = 0;
	double thread_numzeros = 0;
	
	if (thread_id_global == 0){
		*gpu_sum = 0;
		*gpu_abssum = 0;
		*gpu_squaresum = 0;
		*gpu_numzeros = 0;
	}

	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_sum += gpu_x[i];
		thread_abssum += std::abs(gpu_x[i]);
		thread_squaresum += gpu_x[i] * gpu_x[i];
		if (gpu_x[i] == 0)
			thread_numzeros += 1.;	
	}

	
	// now the reduction inside the warp
	int shuffle_delta;
	int lane = threadIdx.x % warpSize;
	for(unsigned int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		if ((lane % (2*stride)) < stride)
			shuffle_delta = stride;
		else
			shuffle_delta = (-1)*stride;
		
		__syncwarp();
		thread_sum += __shfl_down_sync(-1, thread_sum, shuffle_delta);
		thread_abssum += __shfl_down_sync(-1, thread_abssum, shuffle_delta);
		thread_squaresum += __shfl_down_sync(-1, thread_squaresum, shuffle_delta);
		thread_numzeros += __shfl_down_sync(-1, thread_numzeros, shuffle_delta);
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(gpu_sum, thread_sum);
		atomicAdd(gpu_abssum, thread_abssum);
		atomicAdd(gpu_squaresum, thread_squaresum);
		atomicAdd(gpu_numzeros, thread_numzeros);
	}	
}



// ########################################################################
// ########################## Execution wrapper ###########################
// ########################################################################

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


// ########################################################################
// ############################ Toolbox ###################################
// ########################################################################

void cpu_init1(double *vec, size_t size){	
	for(size_t ii = 0; ii < size; ii++)
			vec[ii] = double(ii);
}

void cpu_init2(double *vec, size_t size){	
	for(size_t ii = 0; ii < size; ii++){
			vec[ii] = ii;
			if (ii % 8 == 0)
				vec[ii] = 0.;
			if (ii % 8 == 1)
				vec[ii] = (-1)*double(ii);
			vec[ii] = double(vec[ii]);
	}
}

void print_vector(double *x, size_t size, std::string name = "[vector name]", int width = 4, int prec = 3){	
	std::cout << "printing " + name << std::endl;
	for(size_t i = 0; i < size; i++){
		std::cout.width(width);
		std::cout.precision(prec);
		std::cout << x[i] << ", ";
	}
	std::cout << std::endl;
}


// ########################################################################
// ########################### Main #######################################
// ########################################################################

int main(void)
{
	bool print_results = false;
	Timer global_timer;
	global_timer.reset();

	std::vector<size_t> sizes;
	int numsizes = 12;
	size_t tmp_size = size_t(1e3);

	for(int i = 0; i < numsizes; i += 2){
		sizes.push_back(tmp_size);
		sizes.push_back(3*tmp_size);
		tmp_size *= 10;
	}

	double time_dotp, time_sharedm, time_wshuffle;
	int repetitions = 10;
	
	// print header of csv
	printf("N; time dotp_m[s]; time shared_m[s]; time w_shuffle[s]");	
	if (print_results)
		printf("; results");
	printf("\n");
	
	
	for(int i = 0; i < int(sizes.size()); i++){

		// Allocate vectors on host and device, initialize on host and copy over to device
		double *vec, *gpu_vec;
		vec = (double*)malloc(sizes[i]*sizeof(double));
		cpu_init2(vec, sizes[i]);
		//print_vector(vec,sizes[i]);
		
		cudaMalloc(&gpu_vec, sizes[i]*sizeof(double)); 
		cudaMemcpy(gpu_vec, vec, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);

		// initialize pointers for result
		double *result_sum, *result_abssum, *result_squaresum, *result_numzeros;
		double *gpu_sum, *gpu_abssum, *gpu_squaresum, *gpu_numzeros;
		result_sum = (double*)malloc(sizeof(double));
		result_abssum = (double*)malloc(sizeof(double));
		result_squaresum = (double*)malloc(sizeof(double));
		result_numzeros = (double*)malloc(sizeof(double));

		cudaMalloc(&gpu_sum, sizeof(double));
		cudaMalloc(&gpu_abssum, sizeof(double));
		cudaMalloc(&gpu_squaresum, sizeof(double));
		cudaMalloc(&gpu_numzeros, sizeof(double));
 		
		// Perform the operations
		int grid_size = 256;
		int block_size = BLOCK_SIZE;
		
		time_dotp = execution_wrapper(grid_size, block_size, repetitions, 
																		 gpu_dotproduct_atomicAdd, gpu_vec, gpu_vec, sizes[i],  gpu_sum);		
		time_sharedm = execution_wrapper(grid_size, block_size, repetitions, 
																		 gpu_stuff_sharedm, gpu_vec, sizes[i],  gpu_sum, gpu_abssum, gpu_squaresum, gpu_numzeros);
		time_wshuffle = execution_wrapper(grid_size, block_size, repetitions, 
																		 gpu_stuff_wshuffle, gpu_vec, sizes[i],  gpu_sum, gpu_abssum, gpu_squaresum, gpu_numzeros);
						
		// console output
		printf("%zu;", sizes[i]);
		printf("%1.3e;", time_dotp);
		printf("%1.3e;", time_sharedm);
		printf("%1.3e;", time_wshuffle);
				
		if (print_results){
			cudaMemcpy(result_sum, gpu_sum, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(result_abssum, gpu_abssum, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(result_squaresum, gpu_squaresum, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(result_numzeros, gpu_numzeros, sizeof(double), cudaMemcpyDeviceToHost);			
			printf("%1.5e;",*result_sum);
			printf("%1.5e;",*result_abssum);
			printf("%1.5e;",*result_squaresum);
			printf("%1.5e",*result_numzeros);			
		}
		printf("\n");
		
		
		// cleanup
		free(vec);
		cudaFree(gpu_vec);
		free(result_sum);
		free(result_abssum);
		free(result_squaresum);
		free(result_numzeros);
		cudaFree(gpu_sum);
		cudaFree(gpu_abssum);
		cudaFree(gpu_squaresum);
		cudaFree(gpu_numzeros);		
	}
	
	cudaDeviceReset();  // for CUDA leak checker to work	
	printf("\n total elapsed time in seconds: %2.2f\n", global_timer.get());		
  return EXIT_SUCCESS;
}
 