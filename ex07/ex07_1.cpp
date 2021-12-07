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
void gpu_dotp_wshuffle(const double *x, const double *y, const size_t size, double *dotp){

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
	
	double *host_result, *device_result, *zero;
	zero = (double*)malloc(sizeof(double));	
	*zero = 0;
	host_result = (double*)malloc(sizeof(double));	
	cudaMalloc(&device_result, sizeof(double)); 
	
	cudaMalloc(&device_x, sizes[i]*sizeof(double)); 
	cudaMalloc(&device_y, sizes[i]*sizeof(double)); 
	cudaMemcpy(device_x, host_x, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, host_y, sizes[i]*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(device_result, zero, sizeof(double), cudaMemcpyHostToDevice);
	gpu_dotp_wshuffle<<<1,12>>>(device_x, device_y, sizes[i], device_result);

	cudaMemcpy(host_result, device_result, sizeof(double), cudaMemcpyDeviceToHost);

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




  return EXIT_SUCCESS;
}


