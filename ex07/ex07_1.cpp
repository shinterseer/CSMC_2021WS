#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#include <vector>
#include <map>
#include <cmath>

// #define CUSTOM_SIZE 1000
#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define ALL_MASK 0xffffffff



// ___________________________ changed Kernels ____________________________
// ________________________________________________________________________

__global__
void gpu_dotp_atomic_warp(const double *x, const double *y, const size_t size, double *dotp){

	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	double thread_dotp = 0;
		
	// dot product
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_dotp += x[i] * y[i];
	}
	
	// sumation stage1 (warp reduction)
	for(int stride = warpSize/2; stride > 0; stride /= 2){
		__syncwarp();
		thread_dotp += __shfl_down_sync(ALL_MASK, thread_dotp, stride);			
	}
	__syncwarp();
	
	// sumation stage2 (atomicAdd)
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(dotp, thread_dotp);
	}	
}


__global__
void gpu_dotp_shared(const double *gpu_x, const double *gpu_y, const size_t size, 
										  double *gpu_results){
	
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	__shared__ double shared_dotp[BLOCK_SIZE];
		
	// sum of entries
	double thread_dotp = 0;

	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_dotp += gpu_x[i] * gpu_y[i];
	}

	shared_dotp[threadIdx.x] = thread_dotp;
		
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared_dotp[threadIdx.x] += shared_dotp[threadIdx.x + stride];
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (threadIdx.x == 0){
		gpu_results[blockIdx.x] = shared_dotp[0];
	}	
}

__global__
void gpu_dotp_atomic_shared(const double *gpu_x, const double *gpu_y, const size_t size, 
										  double *gpu_result){
	
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	__shared__ double shared_dotp[BLOCK_SIZE];
		
	// sum of entries
	double thread_dotp = 0;

	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_dotp += gpu_x[i] * gpu_y[i];
	}

	shared_dotp[threadIdx.x] = thread_dotp;
		
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared_dotp[threadIdx.x] += shared_dotp[threadIdx.x + stride];
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_result, shared_dotp[0]);
	}	
}


__global__
void gpu_final_add(double *gpu_block_results, double *gpu_result){
	// we assume one single block 
	// also ideally BLOCK_SIZE == GRID_SIZE (but not necessary)
	__shared__ double shared[GRID_SIZE];
	// __shared__ double shared[gridDim.x];

	shared[threadIdx.x] = gpu_block_results[threadIdx.x];
			
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared[threadIdx.x] += shared[threadIdx.x + stride];
		}
	}

	if (threadIdx.x == 0){
		*gpu_result = shared[0];
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
	std::vector<double> execution_times;		
	
	

	for(int j=0; j<repetitions; j++){
		
		if (device_function){
			cudaDeviceSynchronize(); //make sure gpu is ready
			single_timer.reset();

			function<<<grid_size,block_size>>>(std::forward<ARGS>(arg_pack) ...);

			cudaDeviceSynchronize(); //make sure gpu is done			
			elapsed_time = single_timer.get();
			
		}
		else{
			cudaDeviceSynchronize(); //make sure gpu is ready
			single_timer.reset();

			function(std::forward<ARGS>(arg_pack) ...);

			cudaDeviceSynchronize(); //make sure gpu is done			
			elapsed_time = single_timer.get();			
		}
			
		execution_times.push_back(elapsed_time);
	}
	
	std::sort(execution_times.begin(), execution_times.end());
	median_time = execution_times[int(repetitions/2)];
	return median_time;
}



// __________________________ Timed Programs ______________________________
// ________________________________________________________________________

void cpu_final_add(double *host_block_results, double *host_result, size_t size = GRID_SIZE){
	*host_result = 0;
	for(size_t i = 0; i < size; i++)
		*host_result += host_block_results[i];
}



void final_sum_on_cpu(double *device_x, double *device_y, size_t size, 
											double *device_block_results,	double *host_block_results, double *host_result){
	
	gpu_dotp_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, size, device_block_results);
	cudaMemcpy(host_block_results, device_block_results, GRID_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	cpu_final_add(host_block_results,host_result);
	
}



void final_sum_on_gpu(double *device_x, double *device_y, size_t size, 
											double *device_block_results, double *device_result, double *host_result){
	
	gpu_dotp_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, size, device_block_results);
	gpu_final_add<<<1,GRID_SIZE>>>(device_block_results,device_result);
	cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
	
}

void atomic_add_per_workgroup(double *device_x, double *device_y, int size, double *device_result, double *host_result){
	gpu_dotp_atomic_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, size, device_result);
	cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);	
}

void atomic_add_per_warp(double *device_x, double *device_y, int size, double *device_result, double *host_result){
	gpu_dotp_atomic_warp<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, size, device_result);
	cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
}

// __________________________ Stuff _______________________________________
// ________________________________________________________________________

void cpu_init(double *vec, size_t size){	
	for(size_t ii = 0; ii < size; ii++)
			vec[ii] = double(ii);
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

// _________________________________ Main ________________________________
// ________________________________________________________________________





int main() {

	bool sanity_check = false;
	int repetitions = 11;
	std::vector<size_t> sizes = {size_t(1e3), size_t(3*1e3),
															size_t(1e4), size_t(3*1e4),
															size_t(1e5), size_t(3*1e5),
															size_t(1e6), size_t(3*1e6),
															size_t(1e7)};
	// int i = 0;

	// std::vector<size_t> sizes = {size_t(50)};
	// std::vector<size_t> sizes = {size_t(CUSTOM_SIZE)};

	
	std::cout << "vector size; final_sum_on_cpu; final_sum_on_gpu; atomic_add_per_workgroup; atomic_add_per_warp";
	std::cout << std::endl;
	
	for(size_t i = 0; i < sizes.size(); ++i){
		// Allocate vectors on host and device, initialize on host and copy over to device
		double *host_x, *host_y, *device_x, *device_y;
		host_x = (double*)malloc(sizes[i]*sizeof(double));
		host_y = (double*)malloc(sizes[i]*sizeof(double));
		cpu_init(host_x, sizes[i]);
		cpu_init(host_y, sizes[i]);
		cudaMalloc(&device_x, sizes[i]*sizeof(double)); 
		cudaMalloc(&device_y, sizes[i]*sizeof(double)); 
		cudaMemcpy(device_x, host_x, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(device_y, host_y, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);
		// print_vector(host_x,sizes[i],"host_x");
		// print_vector(host_y,sizes[i],"host_y");
		
		// single number results and initializer zero
		double *host_result, *device_result, *zero;
		zero = (double*)malloc(sizeof(double));	
		*zero = 0;
		host_result = (double*)malloc(sizeof(double));	
		cudaMalloc(&device_result, sizeof(double)); 
		cudaMemcpy(device_result, zero, sizeof(double), cudaMemcpyHostToDevice);

		// multi number results
		double *host_block_results, *device_block_results;
		host_block_results = (double*)malloc(GRID_SIZE*sizeof(double));
		cudaMalloc(&device_block_results, GRID_SIZE*sizeof(double)); 

		double execution_time;
		double sanity_results[4];
		
		std::cout << sizes[i] << "; ";

		// no atomic add. final sum on cpu	
		execution_time = execution_wrapper(GRID_SIZE,
																			BLOCK_SIZE,
																			repetitions,
																			false,
																			final_sum_on_cpu, 
																			device_x, device_y, sizes[i], 
																			device_block_results,	host_block_results, host_result);
		
		std::cout << execution_time << "; ";
		sanity_results[0] = *host_result;

		// no atomic add. final sum on gpu
		execution_time = execution_wrapper(GRID_SIZE,
																			BLOCK_SIZE,
																			repetitions,
																			false,
																			final_sum_on_gpu, 
																			device_x, device_y, sizes[i], 
																			device_block_results, device_result, host_result);
		
		std::cout << execution_time << "; ";
		sanity_results[1] = *host_result;
		
		// atomic add per workgroup
		execution_time = execution_wrapper(GRID_SIZE,
																			BLOCK_SIZE,
																			repetitions,
																			false,
																			atomic_add_per_workgroup, 
																			device_x, device_y, sizes[i], 
																			device_result, host_result);
		
		std::cout << execution_time << "; ";
		sanity_results[2] = *host_result;

		// atomic add once per warp
		execution_time = execution_wrapper(GRID_SIZE,
																			BLOCK_SIZE,
																			repetitions,
																			false,
																			atomic_add_per_warp, 
																			device_x, device_y, sizes[i], 
																			device_result, host_result);
		
		std::cout << execution_time << std::endl;
		sanity_results[3] = *host_result;
		
		if (sanity_check){
			std::cout << std::setprecision(2) 
			<< sanity_results[0] << "; " << sanity_results[1] << "; " 
			<< sanity_results[2] << "; " << sanity_results[3] 
			<< std::endl;
		}

		free(host_x);
		free(host_y);
		free(host_result);
		free(host_block_results);
		free(zero);		
		// free(sanity_results);

		cudaFree(device_x);
		cudaFree(device_y);
		cudaFree(device_result);
		cudaFree(device_block_results);
	}

	cudaDeviceReset();
  return EXIT_SUCCESS;
}


