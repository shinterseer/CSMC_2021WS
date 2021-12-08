#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#include <vector>
#include <map>
#include <cmath>


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


void cpu_final_add(double *host_block_results, double *host_result, size_t size = GRID_SIZE){
	*host_result = 0;
	for(size_t i = 0; i < size; i++)
		*host_result += host_block_results[i];
}


// _________________________________ Main ________________________________
// ________________________________________________________________________

int main() {

	std::vector<size_t> sizes = {100};
	int i = 0;

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


	// no atomic add. final sum on cpu
	// gpu_dotp_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, sizes[i], device_block_results);
	// cudaMemcpy(host_block_results, device_block_results, GRID_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// cpu_final_add(host_block_results,host_result);

	// no atomic add. final sum on gpu
	// gpu_dotp_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, sizes[i], device_block_results);
	// gpu_final_add<<<1,GRID_SIZE>>>(device_block_results,device_result);
	// cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	// atomic add per workgroup
	gpu_dotp_atomic_shared<<<GRID_SIZE,BLOCK_SIZE>>>(device_x, device_y, sizes[i], device_result);
	cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);

	// atomic add once per warp
	// gpu_dotp_wshuffle<<<1,12>>>(device_x, device_y, sizes[i], device_result);
	// cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);

	// std::setprecision(12);
	// std::cout << "result: " << std::setw(20) << *host_result << std::endl;
	// std::cout << "result: " << std::setprecision(12) << *host_result << std::endl;

	// std::cout << "result: " << std::setprecision(12) << std::setw(20) << *host_result << std::endl;
	std::cout << std::setprecision(12) << std::setw(20) 
	<< "result: " << *host_result << std::endl;

	// initialize pointers for result
	// double *result_sum, *result_abssum, *result_squaresum, *result_numzeros;
	// double *gpu_sum, *gpu_abssum, *gpu_squaresum, *gpu_numzeros;
	// result_sum = (double*)malloc(sizeof(double));
	// result_abssum = (double*)malloc(sizeof(double));
	// result_squaresum = (double*)malloc(sizeof(double));
	// result_numzeros = (double*)malloc(sizeof(double));

	// cudaMalloc(&gpu_sum, sizeof(double));
	// cudaMalloc(&gpu_abssum, sizeof(double));
	// cudaMalloc(&gpu_squaresum, sizeof(double));
	// cudaMalloc(&gpu_numzeros, sizeof(double));


	free(host_x);
	free(host_y);
	free(host_result);
	cudaFree(device_x);
	cudaFree(device_y);
	cudaFree(device_result);


  return EXIT_SUCCESS;
}


