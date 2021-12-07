#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
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
void gpu_dotp_wshuffle(const double *x, const double *y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	double thread_dotp = 0;
	
	// initialize result vector for atomicAdd()
	if (thread_id_global == 0){
		dotp = 0;
	}
	
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
		for (int j = 0; j < 8; j++){
			atomicAdd(dotp, thread_dotp);
		}
	}	
}



// _______________________________ Ex4 Kernels ____________________________
// ________________________________________________________________________



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
	// thread 0 of the workgroup writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_result, shared_m[0]);		
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


// _______________________________ Ex2 Kernels ____________________________
// ________________________________________________________________________


__global__
void gpu_dotproduct_stage1(const double *gpu_x, const double *gpu_y, size_t size, double *gpu_result_stage1){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ double shared_m[BLOCK_SIZE];
	
	#ifdef DEBUG
		if (thread_id_global == 0){
			printf("hello from thread 0\n");
			printf("gridDim.x = %i\n", gridDim.x);
			printf("blockDim.x = %i\n", blockDim.x);
		}
		if (thread_id_global == 25*129){
			printf("hello from thread 25*128\n");
			printf("blockIdx.x = %i\n", blockIdx.x);
			printf("threadIdx.x = %i\n", threadIdx.x);
			
		}
	#endif
	
	// lecture sais, this is correct
	/*double thread_dp = 0;
	for (unsigned int i = threadIdx.x; i<size; i += blockDim.x)
		thread_dp += gpu_x[i] * gpu_y[i];
	*/
	
	
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
	
	// thread 0 writes result
	if (threadIdx.x == 0){
		gpu_result_stage1[blockIdx.x] = shared_m[0];
	}	
}


__global__
void gpu_dotproduct_stage2(double *gpu_result_stage1, double *gpu_result_stage2){
	
	// only one block has a job here
	if (blockIdx.x == 0){
		//size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
		__shared__ double shared_m[BLOCK_SIZE];
		
		#ifdef DEBUG
			if (thread_id_global == 0){
				printf("hello from thread 0\n");
				printf("gridDim.x = %i\n", gridDim.x);
				printf("blockDim.x = %i\n", blockDim.x);
			}
			if (thread_id_global == 25*129){
				printf("hello from thread 25*128\n");
				printf("blockIdx.x = %i\n", blockIdx.x);
				printf("threadIdx.x = %i\n", threadIdx.x);
				
			}
		#endif
		
		// this time, the lecture is correct
		double thread_sum = 0;
		for (unsigned int i = threadIdx.x; i<GRID_SIZE; i += blockDim.x)
			thread_sum += gpu_result_stage1[i];
		shared_m[threadIdx.x] = thread_sum;
			
		// now the reduction
		for(int stride = blockDim.x/2; stride>0; stride/=2){
			__syncthreads();
			if (threadIdx.x < stride){
				shared_m[threadIdx.x] += shared_m[threadIdx.x + stride];
			}
		}
		//__syncthreads();
		// thread 0 writes result
		if (threadIdx.x == 0){
			//printf("hi, im thread 0 and im now writing %1.5e\n", shared_m[0]);
			*gpu_result_stage2 = shared_m[0];
		}			
	}
}


__global__
void gpu_dotproduct_atomicAdd2(const double *gpu_x, const double *gpu_y, size_t size, double *gpu_result){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread_id_global == 0)
		*gpu_result = 0;
	
	__shared__ double shared_m[BLOCK_SIZE];
	
	#ifdef DEBUG
		if (thread_id_global == 0){
			printf("hello from thread 0\n");
			printf("gridDim.x = %i\n", gridDim.x);
			printf("blockDim.x = %i\n", blockDim.x);
		}
		if (thread_id_global == 25*129){
			printf("hello from thread 25*128\n");
			printf("blockIdx.x = %i\n", blockIdx.x);
			printf("threadIdx.x = %i\n", threadIdx.x);
			
		}
	#endif
	
	// lecture sais, this is correct
	/*double thread_dp = 0;
	for (unsigned int i = threadIdx.x; i<size; i += blockDim.x)
		thread_dp += gpu_x[i] * gpu_y[i];
	*/
	
	
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
	print_vector(host_x,sizes[i],"host_x");
	print_vector(host_y,sizes[i],"host_y");
	
	// cudaMalloc(&gpu_vec, sizes[i]*sizeof(double)); 
	// cudaMemcpy(gpu_vec, vec, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);

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




  return EXIT_SUCCESS;
}


