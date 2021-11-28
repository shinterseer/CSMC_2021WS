#include <stdio.h>
#include <iostream>
#include "timer.hpp"
#include <vector>
#include <algorithm>

#define BLOCK_DIM 16


// ########################################################################
// ################## Original Kernel #####################################
// ########################################################################

__global__
void transpose(double *A, int N)
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row_idx = t_idx / N;
  int col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N) A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
}


// ########################################################################
// ################## The Kernels #########################################
// ########################################################################

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


__global__
void transpose_correct_opt1(const double *A, double *B, int N)
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
// this is the non-inplace version of the optimization
{
	// split the whole matrix in 16x16 blocks
	int global_row_idx, global_col_idx; // the index of the block on the coarser grid
	int local_row_idx, local_col_idx; // the index within the block
	int num_blocks_x = N / BLOCK_DIM;
	
	double temp_for_swapping;	
	__shared__
	double block[BLOCK_DIM * BLOCK_DIM];		
		
	// iterate over the blocks (iterate over workgroups/blocks)
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
		
		__syncthreads(); // make sure, the bloc is loaded before transposing
				
		// transpose the block (iterate over threads within workgroup)
		for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = j / BLOCK_DIM;
			local_col_idx = j % BLOCK_DIM;
			temp_for_swapping = block[local_col_idx*BLOCK_DIM + local_row_idx];
			__syncthreads(); // make sure, everybody has grabbed their entry before swapping
			block[local_row_idx*BLOCK_DIM + local_col_idx] = temp_for_swapping;			
		}

		__syncthreads(); // make sure, everything is transposed before writing result
		// write the transposed block (iterate over threads within workgroup)
		for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
			local_row_idx = j / BLOCK_DIM;
			local_col_idx = j % BLOCK_DIM;

			// now we write to the mirrored block position => global_row_idx and global_col_idx are swapped
			B[(global_col_idx*BLOCK_DIM + local_row_idx)*N + (global_row_idx*BLOCK_DIM + local_col_idx)] =
																												block[local_row_idx*BLOCK_DIM + local_col_idx];
		}
	}	
}


__global__
void transpose_correct_opt2_inplace(double *A, int N)
// this is the inplace version of the optimization. something is wrong, but I didnt find the mistake
{
	// split the whole matrix in 16x16 blocks
	int global_row_idx, global_col_idx; // the index of the block on the coarser grid
	int local_row_idx, local_col_idx; // the index within the block
	int num_blocks_x = N / BLOCK_DIM;
	
	double temp_for_swapping;	
	__shared__
	double block_ij[BLOCK_DIM * BLOCK_DIM];	
	__shared__
	double block_ji[BLOCK_DIM * BLOCK_DIM];		
	
	// iterate over the blocks (iterate over workgroups/blocks)
	for(int i = blockIdx.x; i < num_blocks_x*num_blocks_x; i += gridDim.x){
		global_row_idx = i / num_blocks_x;
		global_col_idx = i % num_blocks_x;
		
		// only iterate over upper triangular matrix, because, every workgoup will deal with both block_ij and block_ji
		// for blocks on the diagonal, the transposition is redundantly duplicated - should still work though
		if (global_col_idx >= global_row_idx)
		{
			// load block_ij and block_ji into shared objects (iterate over threads within workgroup)
			for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j +=blockDim.x){
				local_row_idx = j / BLOCK_DIM;
				local_col_idx = j % BLOCK_DIM;
				
				block_ij[local_row_idx*BLOCK_DIM + local_col_idx] = A[(global_row_idx*BLOCK_DIM + local_row_idx)*N + 
																										(global_col_idx*BLOCK_DIM + local_col_idx)];

				// swap global indices to access mirrored block
				block_ji[local_row_idx*BLOCK_DIM + local_col_idx] = A[(global_col_idx*BLOCK_DIM + local_row_idx)*N + 
																										(global_row_idx*BLOCK_DIM + local_col_idx)];
			}
			
			__syncthreads(); // make sure, the blocks are loaded before transposing
					
			// transpose the blocks (iterate over threads within workgroup)
			for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
				local_row_idx = j / BLOCK_DIM;
				local_col_idx = j % BLOCK_DIM;
				//transpose block_ij
				temp_for_swapping = block_ij[local_col_idx*BLOCK_DIM + local_row_idx];
				__syncthreads(); // make sure, everybody has grabbed their entry before swapping
				block_ij[local_row_idx*BLOCK_DIM + local_col_idx] = temp_for_swapping;			

				//transpose block_ji
				temp_for_swapping = block_ji[local_col_idx*BLOCK_DIM + local_row_idx];
				__syncthreads(); // make sure, everybody has grabbed their entry before swapping
				block_ji[local_row_idx*BLOCK_DIM + local_col_idx] = temp_for_swapping;			
			}

			__syncthreads(); // make sure, everything is transposed before writing result
			// write the transposed block (iterate over threads within workgroup)
			for(int j = threadIdx.x; j < BLOCK_DIM * BLOCK_DIM; j += blockDim.x){
				local_row_idx = j / BLOCK_DIM;
				local_col_idx = j % BLOCK_DIM;

				// now we write to the mirrored block position => global_row_idx and global_col_idx are swapped
				// write block_ij			
				A[(global_col_idx*BLOCK_DIM + local_row_idx)*N + (global_row_idx*BLOCK_DIM + local_col_idx)] =
																													block_ij[local_row_idx*BLOCK_DIM + local_col_idx];
				// write block_ji
				A[(global_row_idx*BLOCK_DIM + local_row_idx)*N + (global_col_idx*BLOCK_DIM + local_col_idx)] =
																													block_ji[local_row_idx*BLOCK_DIM + local_col_idx];
			}			
		} // if upper triangular
	}	// for loop over blocks
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
// ######################## Helper Functions ##############################
// ########################################################################

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

	std::vector<int> sizes{512, 1024, 2048, 4096};
	//std::vector<int> sizes{16};
  double *A, *cuda_A, *cuda_B;
	bool check_results = false;
	bool print_to_screen = false;

	double elapsed_time;
	int repetitions = 11;
	
	printf("\n");
	printf("transposing matrix \n");
	printf("----------------------------------------- \n");	
	printf("repetitions = %i \n",repetitions);
	printf("N; median elapsed time in seconds\n");
	
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
		int num_workgroups = sizes[i] / BLOCK_DIM * sizes[i] / BLOCK_DIM;

		//elapsed_time = execution_wrapper(num_workgroups, BLOCK_DIM*BLOCK_DIM, repetitions, 
		//																 transpose_correct, cuda_A, sizes[i]);
		//double* (&result) = cuda_A;

		//elapsed_time = execution_wrapper(num_workgroups, BLOCK_DIM*BLOCK_DIM, repetitions, 
		//																 transpose_correct_opt1, cuda_A, cuda_B, sizes[i]);
		//double* (&result) = cuda_B;
		
		//elapsed_time = execution_wrapper(num_workgroups, BLOCK_DIM*BLOCK_DIM, repetitions, 
		//																 transpose_correct_opt2, cuda_A, cuda_B, sizes[i]);
		//double* (&result) = cuda_B;
		
		elapsed_time = execution_wrapper(num_workgroups, BLOCK_DIM*BLOCK_DIM, repetitions, 
																		 transpose_correct_opt2_inplace, cuda_A, sizes[i]);
		double* (&result) = cuda_A;
				
		printf("%i;", sizes[i]);
		printf("%1.3e\n", elapsed_time);

		
		if (check_results){
			double *AT;
			AT = (double*)malloc(sizes[i]*sizes[i]*sizeof(double));
			// copy data back (implicit synchronization point)
			cudaMemcpy(AT, result, sizes[i]*sizes[i]*sizeof(double), cudaMemcpyDeviceToHost);
			printf("result is correct: %s \n",check_correctness(A,AT,sizes[i]) ? "true" : "false");
			if (print_to_screen)
				print_A(AT, sizes[i]);
			free(AT);			
		}
		
		free(A);
		cudaFree(cuda_A);
		cudaFree(cuda_B);
		cudaDeviceReset();  // for CUDA leak checker to work
	}
	   
	printf("\n total elapsed time in seconds: %2.2f\n", global_timer.get());		
  return EXIT_SUCCESS;
}
 