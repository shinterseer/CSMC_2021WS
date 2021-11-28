

#include <stdio.h>
#include <iostream>
#include "timer.hpp"
#include <vector>
#include <algorithm>

#define BLOCK_DIM 16

__global__
void transpose(double *A, int N)
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row_idx = t_idx / N;
  int col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N) A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
}


__global__
void transpose_correct(double *A, int N)
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = gridDim.x * blockDim.x;
  int row_idx;
  int col_idx;
  double temp;

	for(size_t i = t_idx; i < N*N; i += num_threads){
		row_idx = i / N;
		col_idx = i % N;
		
		if (row_idx < N && col_idx > row_idx){
			temp = A[row_idx * N + col_idx];
			A[row_idx * N + col_idx] = A[col_idx * N + row_idx];	
			A[col_idx * N + row_idx] = temp;
		}		
	}
}


// ########################################################################
// ################## The Kernel ##########################################
// ########################################################################

__global__
void transpose_correct_opt1(double *A, double *B, int N)
// this is the non-inplace version of the optimization
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = gridDim.x * blockDim.x;
  int row_idx;
  int col_idx;
  //double temp;

	for(size_t i = t_idx; i < N*N; i += num_threads){
		row_idx = i / N;
		col_idx = i % N;
		
		if (row_idx < N){
			B[col_idx * N + row_idx] = A[row_idx * N + col_idx];
		}		
	}
}


__global__
void transpose_correct_opt2(const double *A, double *B, int N)
// this is the inplace version of the optimization
{


	// split the whole matrix in 16x16 blocks
	int global_row_idx, global_col_idx;
	int local_row_idx, local_col_idx;
	int num_blocks_x = N / BLOCK_DIM;
  //int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int num_threads = gridDim.x * blockDim.x;
	
	// declare shared 16x16 block
	__shared__
	double block[BLOCK_DIM * BLOCK_DIM];
	__shared__
	double blockT[BLOCK_DIM * BLOCK_DIM];
	
		
	// iterate over the blocks (iterate over workgroups)
	for(int i = blockIdx.x; i < num_blocks_x*num_blocks_x; i += gridDim.x){
		global_row_idx = i / num_blocks_x;
		global_col_idx = i % num_blocks_x;
		
		// load one block into a shared object (iterate over threads within workgroup)
		for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j +=blockDim.x){
			local_row_idx = j / BLOCK_DIM;
			local_col_idx = j % BLOCK_DIM;
			
			block[local_row_idx*BLOCK_DIM + local_col_idx] = A[(global_row_idx*BLOCK_DIM + local_row_idx)*N + 
																									(global_col_idx*BLOCK_DIM + local_col_idx)];
		}
		
		__syncthreads(); // make sure, everything is loaded
		// transpose the block (iterate over threads within workgroup)
		for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = j / BLOCK_DIM;
			local_col_idx = j % BLOCK_DIM;
			blockT[local_row_idx*BLOCK_DIM + local_col_idx] = block[local_col_idx*BLOCK_DIM + local_row_idx];			
		}

		__syncthreads(); // make sure, everything is transposed
		// write the transposed block (iterate over threads within workgroup)
		for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = j / BLOCK_DIM;
			local_col_idx = j % BLOCK_DIM;

			// now we write to the mirrored block position => global_row_idx and global_col_idx are swapped
			B[(global_col_idx*BLOCK_DIM + local_row_idx)*N + (global_row_idx*BLOCK_DIM + local_col_idx)] =
																												blockT[local_row_idx*BLOCK_DIM + local_col_idx];
		}
	}	
}

__global__
void transpose_correct_opt3(const double *A, double *B, int N)
// this is the inplace version of the optimization
{


	// split the whole matrix in 16x16 blocks
	int global_row_idx, global_col_idx;
	int local_row_idx, local_col_idx;
	int num_blocks_x = N / BLOCK_DIM;
  //int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int num_threads = gridDim.x * blockDim.x;
	
	double temp_for_swapping;	
	// declare shared 16x16 block
	__shared__
	double block[BLOCK_DIM * BLOCK_DIM];	
	//__shared__
	//double blockT[BLOCK_DIM * BLOCK_DIM];
	
		
	// iterate over the blocks (iterate over workgroups)
	for(int i = blockIdx.x; i < num_blocks_x*num_blocks_x; i += gridDim.x){
		global_row_idx = i / num_blocks_x;
		global_col_idx = i % num_blocks_x;
		
		// load one block into a shared object (iterate over threads within workgroup)
		//for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j +=blockDim.x){
			local_row_idx = threadIdx.x / BLOCK_DIM;
			local_col_idx = threadIdx.x % BLOCK_DIM;
			
			block[local_row_idx*BLOCK_DIM + local_col_idx] = A[(global_row_idx*BLOCK_DIM + local_row_idx)*N + 
																									(global_col_idx*BLOCK_DIM + local_col_idx)];
		//}
		
		__syncthreads(); // make sure, everything is loaded before transposing
		
		
		
		// transpose the block (iterate over threads within workgroup)
		//for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = threadIdx.x / BLOCK_DIM;
			local_col_idx = threadIdx.x % BLOCK_DIM;
			temp_for_swapping = block[local_col_idx*BLOCK_DIM + local_row_idx];
		__syncthreads(); // make sure, everything is loaded before transposing
			temp_for_swapping = block[local_row_idx*BLOCK_DIM + local_col_idx] = temp_for_swapping;
			
			
//			blockT[local_row_idx*BLOCK_DIM + local_col_idx] = block[local_col_idx*BLOCK_DIM + local_row_idx];
		//}

		__syncthreads(); // make sure, everything is transposed before writing result
		// write the transposed block (iterate over threads within workgroup)
		//for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = threadIdx.x / BLOCK_DIM;
			local_col_idx = threadIdx.x % BLOCK_DIM;

			// now we write to the mirrored block position => global_row_idx and global_col_idx are swapped
			B[(global_col_idx*BLOCK_DIM + local_row_idx)*N + (global_row_idx*BLOCK_DIM + local_col_idx)] =
																												block[local_row_idx*BLOCK_DIM + local_col_idx];
		//}
	}	
}


// ########################################################################
// ################## The very cool execution wrapper #####################
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


void print_A(double *A, int N)
{
  for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; ++j) {
		std::cout << A[i * N + j] << ", ";
	}
	std::cout << std::endl;
  }
}


// checks, if AT is the transposed matrix of A
bool check_correctness(double* A, double* AT, int N){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if (A[i*N + j] != AT[j*N + i]){
				printf("error at row = %i, col = %i\n",i,j);
				return false;				
			}
		}
	}
	return true;	
}


// ########################################################################
// ########################### Main #######################################
// ########################################################################

int main(void)
{
	Timer global_timer;
	global_timer.reset();

	//std::vector<int> sizes{512, 1024, 2048, 4096};
	std::vector<int> sizes{16};
  double *A, *cuda_A, *cuda_B;
	bool check_results = true;
	bool print_to_screen = false;

	double elapsed_time;
	int repetitions = 10;
	
	printf("\n");
	printf("transposing matrix \n");
	printf("----------------------------------------- \n");	
	printf("repetitions = %i \n",repetitions);
	printf("N; median elapsed time in seconds;\n");
	
	for(int i = 0; i < int(sizes.size()); i++){
		
		// Allocate host memory and initialize
		A = (double*)malloc(sizes[i]*sizes[i]*sizeof(double));
		//A_original = (double*)malloc(sizes[i]*sizes[i]*sizeof(double));
		
		for (int j = 0; j < sizes[i]*sizes[i]; j++) {
			A[j] = j;
		}
		if (print_to_screen)
			print_A(A, sizes[i]);

		// Allocate device memory and copy host data over
		cudaMalloc(&cuda_A, sizes[i]*sizes[i]*sizeof(double)); 
		cudaMalloc(&cuda_B, sizes[i]*sizes[i]*sizeof(double)); 
 
		// copy data over
		cudaMemcpy(cuda_A, A, sizes[i]*sizes[i]*sizeof(double), cudaMemcpyHostToDevice);
 		
		// Perform the transpose operation
		//elapsed_time = execution_wrapper((sizes[i]+255)/256, BLOCK_DIM*BLOCK_DIM, repetitions, transpose_correct_opt3, cuda_A, cuda_B, sizes[i]);
		//elapsed_time = execution_wrapper(256, BLOCK_DIM*BLOCK_DIM, repetitions, transpose_correct_opt1, cuda_A, cuda_B, sizes[i]);
		//elapsed_time = execution_wrapper(256, BLOCK_DIM*BLOCK_DIM, repetitions, transpose_correct_opt2, cuda_A, cuda_B, sizes[i]);
		elapsed_time = execution_wrapper(256, BLOCK_DIM*BLOCK_DIM, repetitions, transpose_correct_opt3, cuda_A, cuda_B, sizes[i]);
				
		printf("%i;", sizes[i]);
		printf("%1.3e\n", elapsed_time);

		
		if (check_results){
			double *AT;
			AT = (double*)malloc(sizes[i]*sizes[i]*sizeof(double));
			// copy data back (implicit synchronization point)
			cudaMemcpy(AT, result, sizes[i]*sizes[i]*sizeof(double), cudaMemcpyDeviceToHost);
			printf("result is correct: %s \n",check_correctness(A,AT,sizes[i]) ? "true" : "false");
			//bool correct = check_correctness(A,AT,sizes[i]);
			//printf("result is correct: %i \n",int(correct));			
			if (print_to_screen)
				print_A(AT, sizes[i]);
			free(AT);			
		}
		
		free(A);
		//free(A_original);
		
		cudaFree(cuda_A);
		cudaFree(cuda_B);
		cudaDeviceReset();  // for CUDA leak checker to work
	}
	 
  //std::cout << std::endl << "Time for transpose: " << elapsed_time << std::endl;
  //std::cout << "Effective bandwidth: " << (2*N*N*sizeof(double)) / elapsed_time * 1e-9 << " GB/sec" << std::endl;
  //std::cout << std::endl;
  
	printf("\n total elapsed time in seconds: %2.2f\n", global_timer.get());		
  return EXIT_SUCCESS;
}
 

