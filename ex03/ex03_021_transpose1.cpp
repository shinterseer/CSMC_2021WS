

#include <stdio.h>
#include <iostream>
#include "timer.hpp"
 
 
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
			if (A[i*N + j] != AT[j*N + i])
				return false;
		}
	}
	return true;	
}


// ########################################################################
// ########################### Main #######################################
// ########################################################################

 
int main(void)
{
  int N = 4096;
 
  double *A, *cuda_A;
  double *A_original;
  Timer timer;
 
  // Allocate host memory and initialize
  A = (double*)malloc(N*N*sizeof(double));
  A_original = (double*)malloc(N*N*sizeof(double));
  
  for (int i = 0; i < N*N; i++) {
	A[i] = i;
	A_original[i] = i;
  }
 
	if (N <= 20)
		print_A(A, N);
 
 
  // Allocate device memory and copy host data over
  cudaMalloc(&cuda_A, N*N*sizeof(double)); 
 
  // copy data over
  cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
 
  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();
 
  // Perform the transpose operation
  //transpose<<<(N+255)/256, 256>>>(cuda_A, N);
  //transpose_correct<<<(N+255)/256, 256>>>(cuda_A, N);
  transpose_correct<<<256, 256>>>(cuda_A, N);
 
  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  double elapsed = timer.get();
  std::cout << std::endl << "Time for transpose: " << elapsed << std::endl;
  std::cout << "Effective bandwidth: " << (2*N*N*sizeof(double)) / elapsed * 1e-9 << " GB/sec" << std::endl;
  std::cout << std::endl;

  // copy data back (implicit synchronization point)
  cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);

	if (N <= 20)
		print_A(A, N);
	
	printf("result is correct: %s \n",check_correctness(A,A_original,N) ? "true" : "false");
	
	free(A);
	free(A_original);
	cudaFree(cuda_A);
  cudaDeviceReset();  // for CUDA leak checker to work
 
  return EXIT_SUCCESS;
}
 

