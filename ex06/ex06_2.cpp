#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define GRID_SIZE 256
#define BLOCK_SIZE 256



// ___________________________ Device Kernels _____________________________
// ________________________________________________________________________


void assembleA(int *csr_rowoffsets, double *csr_values, int* csr_colindices, int N, int M) {
//void assembleA(double *csr_rowoffsets, double *csr_values, int* csr_colindices, int N, int M) {
	
	for (int row = 0;	
 			 row < N*M;
			 ++row) {
				 
			int i = row / M;
			int j = row % M;
			int this_row_offset = csr_rowoffsets[row];

			int index = i * M + j;


			// bottom side (S)
			if (i > 0){
				csr_colindices[this_row_offset] = index - M;
				csr_values[this_row_offset] = -1;				
				this_row_offset += 1;
			};

			// left side (W)
			if (j > 0) {
				csr_colindices[this_row_offset] = index - 1;
				csr_values[this_row_offset] = -1;
				this_row_offset += 1;
			}

			// diagonal element (C)
			csr_colindices[this_row_offset] = index;
			csr_values[this_row_offset] = 4;
			this_row_offset += 1;
			
			// right side (E)
			if (j < M-1){
				csr_colindices[this_row_offset] = index + 1;
				csr_values[this_row_offset] = -1;
				this_row_offset += 1;
			}

			// top side (N)
			if (i < M-1){
				csr_colindices[this_row_offset] = index + M;
				csr_values[this_row_offset] = -1;				
				this_row_offset += 1;
			}
			
	}
}


//__global__
void num_nonzero_entries(int N, int M, int *output){
//void num_nonzero_entries(int N, int M, double *output){
	// inidices and dimensions of physical grid: 
	// i ... row idx, j ... col idx
	// N ... row dim, M ... col dim
	// adressing: grid(i,j)

	// system matrix A is of dimensions N*M x N*M
	// row (i*M + j) represents the gridpoint grid[i,j]

	
	for (size_t i = 0; i < N; i++){
		for (size_t j = 0; j < M; j++){			
			// how many nonzero entries does row i*M + j have
			output[i*M + j] = 0;
			
			// diagonal element (C)
			output[i*M + j]++;
			// left side (W)
			if (i > 0) output[i*M + j]++;
			// right side (E)
			if (i < N-1) output[i*M + j]++;
			// bottom side (S)
			if (j > 0) output[i*M + j]++;
			// top side (N)
			if (j < M-1) output[i*M + j]++;
		}		
	}
}




__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ int shared_buffer[BLOCK_SIZE];
  int my_value;

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



// exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[GRID_SIZE];

  // load data:
  int my_carry = carries[threadIdx.x];

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

__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;

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

void exclusive_scan(int const * input,
                    int       * output, int N)
{

  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRID_SIZE);

  // First step: Scan within each thread group and write carries
  //scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);
  scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);

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
	
	int N = 3;
	int M = 3;
	bool sanity_check = true;
	
	int *num_entries = (int *)malloc(sizeof(int) * N*M);
	int *csr_rowoffsets = (int *)malloc(sizeof(int) * N*M);
	
	int* device_num_entries;
	cudaMalloc(&device_num_entries, sizeof(int) * N*M);
	
	int* device_csr_rowoffsets;
	cudaMalloc(&device_csr_rowoffsets, sizeof(int) * N*M);
	

	// get number of nonzero entries per row
	num_nonzero_entries(N, M, num_entries);

	//if (sanity_check){
	if (false){
		for(int i = 0; i < N*M; ++i){
			std::cout << num_entries[i] << ", ";
		}
		std::cout << std::endl;
	}

	cudaMemcpy(device_num_entries, num_entries, sizeof(int) * N*M, cudaMemcpyHostToDevice);	
	
	// get row offsets for CSR as exclusive prefix sum of num_entries	
	exclusive_scan(device_num_entries, device_csr_rowoffsets, N*M);
		
	cudaMemcpy(csr_rowoffsets, device_csr_rowoffsets, sizeof(int) * N*M, cudaMemcpyDeviceToHost);
	csr_rowoffsets[N*M] = csr_rowoffsets[N*M - 1] + num_entries[N*M - 1];
	
	
	int *csr_colindices = (int *)malloc(sizeof(int) * csr_rowoffsets[N*M]);
	double *csr_values = (double *)malloc(sizeof(double) * csr_rowoffsets[N*M]);

	assembleA(csr_rowoffsets, csr_values, csr_colindices, N, M);
	//get_system_matrix(csr_rowoffsets, csr_values, csr_colindices, N, M);

	if (sanity_check){
		for(int i = 0; i < N*M; ++i){
			//std::cout << num_entries[i] << ", ";
		}
		std::cout << std::endl;
		for(int i = 0; i < N*M+1; ++i){
			std::cout << csr_rowoffsets[i] << ", ";
		}
		std::cout << std::endl;
	}


	if (sanity_check){
		std::cout << "csr_colindices: " << std::endl;
		for(int i = 0; i < csr_rowoffsets[N*M]; ++i){
			std::cout << csr_colindices[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "csr_values: " << std::endl;
		for(int i = 0; i < csr_rowoffsets[N*M]; ++i){
			std::cout << csr_values[i] << ", ";
		}
		std::cout << std::endl;
		
		for(int i = 1; i < 10; ++i){
			csr_values[i] = 0;
		}
		
		generate_fdm_laplace(N, csr_rowoffsets, csr_colindices, csr_values);
		
		
		std::cout << "correct values:  " << std::endl;
		
		std::cout << "csr_colindices: " << std::endl;
		for(int i = 0; i < csr_rowoffsets[N*M]; ++i){
			std::cout << csr_colindices[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "csr_values: " << std::endl;
		for(int i = 0; i < csr_rowoffsets[N*M]; ++i){
			std::cout << csr_values[i] << ", ";
		}
		std::cout << std::endl;
		
			
		
	}



  return EXIT_SUCCESS;
}


