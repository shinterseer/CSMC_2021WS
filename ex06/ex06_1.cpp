#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define GRID_SIZE 256
#define BLOCK_SIZE 256



// ___________________________ Device Kernels _____________________________
// ________________________________________________________________________


__global__ void scan_kernel_1(double const *X,
                              double *Y,
                              int N,
                              double *carries)
{
  __shared__ double shared_buffer[BLOCK_SIZE];
  double my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}

__global__ void scan_kernel_1_incl(double const *X,
                              double *Y,
                              int N,
                              double *carries)
{
  __shared__ double shared_buffer[BLOCK_SIZE];
  double my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    //my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}




// exclusive-scan of carries
__global__ void scan_kernel_2(double *carries)
{
  __shared__ double shared_buffer[GRID_SIZE];

  // load data:
  double my_carry = carries[threadIdx.x];

  // exclusive scan in shared buffer:

  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}

__global__ void scan_kernel_3(double *Y, int N,
                              double const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ double shared_offset;

  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}


__global__
void device_excl_to_incl(const double *device_input, double *device_output, int N)
{
	int num_threads = blockDim.x * gridDim.x;
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	for(size_t i = thread_id; i < N; i += num_threads)
		device_output[i] += device_input[i];	
}




// ____________________________ Host Programs _____________________________
// ________________________________________________________________________

void exclusive_scan(double const * input,
                    double       * output, int N)
{

  double *carries;
  cudaMalloc(&carries, sizeof(double) * GRID_SIZE);

  // First step: Scan within each thread group and write carries
  //scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);
  scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, GRID_SIZE>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<GRID_SIZE, BLOCK_SIZE>>>(output, N, carries);

  cudaFree(carries);
}


void inclusive_scan1(const double *device_input, double *device_output, int N)
{
	// get exclusive scan
  exclusive_scan(device_input, device_output, N);

	// add input to output
	device_excl_to_incl<<<GRID_SIZE, BLOCK_SIZE>>>(device_input, device_output, N);
}

void inclusive_scan2(double const * input,
                    double       * output, int N)
{

  double *carries;
  cudaMalloc(&carries, sizeof(double) * GRID_SIZE);

  // First step: Scan within each thread group and write carries
  //scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);
  scan_kernel_1_incl<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, GRID_SIZE>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<GRID_SIZE, BLOCK_SIZE>>>(output, N, carries);

  cudaFree(carries);
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



// _________________________________ Main ________________________________
// ________________________________________________________________________

int main() {

  //int N = 2000;
	std::vector<size_t> N = {size_t(1e4), size_t(3e4), 
													 size_t(1e5), size_t(3e5), 
													 size_t(1e6), size_t(3e6), 
													 size_t(1e7), size_t(3e7), 
													 size_t(1e8), size_t(3e8), 
													};
	
	
	bool sanity_check = false;
	double execution_time = 0;
	int repetitions = 11;

	std::cout << "N; elapsed time in s" << std::endl;

  // Perform the exclusive scan and obtain results


	for(int ii = 0; ii < N.size(); ++ii){

		//
		// Allocate host arrays for reference
		//
		double *x = (double *)malloc(sizeof(double) * N[ii]);
		double *y = (double *)malloc(sizeof(double) * N[ii]);
		double *z = (double *)malloc(sizeof(double) * N[ii]);
		
		
		std::fill(x, x +N[ii], 1);

		//
		// Allocate CUDA-arrays
		//
		double *cuda_x, *cuda_y;
		cudaMalloc(&cuda_x, sizeof(double) * N[ii]);
		cudaMalloc(&cuda_y, sizeof(double) * N[ii]);
		cudaMemcpy(cuda_x, x, sizeof(double) * N[ii], cudaMemcpyHostToDevice);


		// reference calculation:
		if (sanity_check){
			y[0] = 0;
			for (std::size_t i=1; i<N[ii]; ++i) y[i] = y[i-1] + x[i-1];		
		}

		execution_time = execution_wrapper(GRID_SIZE, BLOCK_SIZE, repetitions, false, 
																			 inclusive_scan2, cuda_x, cuda_y, N[ii]);

		printf("%i;",N[ii]);
		printf("%5.8e\n",execution_time);
		
			cudaMemcpy(z, cuda_y, sizeof(double) * N[ii], cudaMemcpyDeviceToHost);
	
		//
		// Print first few entries for reference
		//
		if (sanity_check){
			std::cout << "CPU y: ";
			for (int i=0; i<10; ++i) std::cout << y[i] << " ";
			std::cout << " ... ";
			for (int i=N[ii]-10; i<N[ii]; ++i) std::cout << y[i] << " ";
			std::cout << std::endl;

			std::cout << "GPU y: ";
			for (int i=0; i<10; ++i) std::cout << z[i] << " ";
			std::cout << " ... ";
			for (int i=N[ii]-10; i<N[ii]; ++i) std::cout << z[i] << " ";
			std::cout << std::endl;		
		}

		//
		// Clean up:
		//
		free(x);
		free(y);
		free(z);
		cudaFree(cuda_x);
		cudaFree(cuda_y);		
	}
  return EXIT_SUCCESS;
}

