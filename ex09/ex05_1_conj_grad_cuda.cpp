#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"

# define GRID_SIZE 256
# define BLOCK_SIZE 256

 
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
 
// __global__
// void gpu_dotp_wshuffle(const double *x, const double *y, const size_t size, double *dotp){
		
	// int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	// int thread_num = gridDim.x * blockDim.x;
	
	// sum of entries
	// double thread_dotp = 0;
	
	// if (thread_id_global == 0){
		// *dotp = 0;
	// }
 
	// for (unsigned int i = thread_id_global; i < size; i += thread_num){
		// thread_dotp += x[i] * y[i];
	// }
 
	
	// now the reduction inside the warp
	// int shuffle_delta;
	// int lane = threadIdx.x % warpSize;
	// for(int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		// if ((lane % (2*stride)) < stride)
			// shuffle_delta = stride;
		// else
			// shuffle_delta = (-1)*stride;
		
		// __syncwarp();
		// thread_dotp += __shfl_down_sync(-1, thread_dotp, shuffle_delta);
	// }
	
	// __syncwarp();
	// thread 0 (of each warp) writes result
	// if ((threadIdx.x % warpSize) == 0){
		// atomicAdd(dotp, thread_dotp);
	// }	
// }
 
__global__
void gpu_line6and7(const size_t size, double *x, 
									 const double* res_norm, const double* pAp, double *alpha, const double *p){
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	*alpha = *res_norm / *pAp;
	//if (thread_id_global == 0) printf("alpha: %3.10f\n",*alpha);
	for (size_t i = thread_id_global; i < size; i += thread_num){
		x[i] += (*alpha) * p[i];
	}
}
 
 
 
__global__
void gpu_line8(const size_t size, double *r, const double *alpha, const double *Ap){	
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	for (size_t i = thread_id_global; i < size; i += thread_num){
		r[i] -= (*alpha) * Ap[i];		
	}
}
 
 
__global__
void gpu_line12(const size_t size, double *p, const double *r, 
								const double* res_norm_old, const double* res_norm_new){		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	double beta = (*res_norm_new) / (*res_norm_old);
	for (size_t i = thread_id_global; i < size; i += thread_num){
		p[i] = r[i] + beta * p[i];		
	}
}
 
 
/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
 
__global__
void gpu_csr_matvec_product_par(const size_t N,
														const int *csr_rowoffsets, const int *csr_colindices, const double *csr_values,
														const double *x, double *y)
{
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	size_t num_threads = gridDim.x * blockDim.x;
	
	for (size_t row = thread_id_global; row < N; row += num_threads) {
		double val = 0; // y = Ax for this row
		for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row+1]; ++jj) {
			val += csr_values[jj] * x[csr_colindices[jj]];
		}
		y[row] = val;
	}	
}
 
 
/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 */
void gpu_conjugate_gradient(size_t N,  // number of unknows
												int *csr_rowoffsets, int *csr_colindices, double *csr_values,
												double *rhs,
												double *solution,
												int max_iters = 100)
												//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
 
	// clear solution vector (it may contain garbage values):
	std::fill(solution, solution + N, 0);
	
	// Allocate on gpu
	double *gpu_solution;
	cudaMalloc(&gpu_solution, N * sizeof(double));
	int *gpu_csr_rowoffsets, *gpu_csr_colindices;	
	cudaMalloc(&gpu_csr_rowoffsets, (N+1) * sizeof(int));
	cudaMalloc(&gpu_csr_colindices, 5*N * sizeof(int));
	double *gpu_csr_values;
	cudaMalloc(&gpu_csr_values, 5*N * sizeof(double));
	// copy over to gpu
	cudaMemcpy(gpu_solution, solution, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_csr_rowoffsets, csr_rowoffsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_csr_colindices, csr_colindices, 5*N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_csr_values, csr_values, 5*N * sizeof(double), cudaMemcpyHostToDevice);
 
	
	
	// initialize work vectors:
	double *r = (double*)malloc(sizeof(double) * N);
	double *p = (double*)malloc(sizeof(double) * N);
	double *Ap = (double*)malloc(sizeof(double) * N);
 
	double *gpu_p, *gpu_Ap;
	cudaMalloc(&gpu_p, N * sizeof(double));
	cudaMalloc(&gpu_Ap, N * sizeof(double));
	double *gpu_r, *gpu_alpha, *gpu_beta, *gpu_pAp;
	cudaMalloc(&gpu_r, N * sizeof(double));
	//cudaMalloc(&gpu_res_norm, sizeof(double));
	cudaMalloc(&gpu_alpha, sizeof(double));
	cudaMalloc(&gpu_beta, sizeof(double));
	cudaMalloc(&gpu_pAp, sizeof(double));
	double *gpu_res_norm_old, *gpu_res_norm_new;
	cudaMalloc(&gpu_res_norm_old, sizeof(double));
	cudaMalloc(&gpu_res_norm_new, sizeof(double));
	
	// line 2: initialize r and p:
	std::copy(rhs, rhs+N, p);
	std::copy(rhs, rhs+N, r);
	cudaMemcpy(gpu_p, p, sizeof(double) * N, cudaMemcpyHostToDevice);		
	cudaMemcpy(gpu_r, r, sizeof(double) * N, cudaMemcpyHostToDevice);
 
	int iters = 0;
	while (1) {
 
		// line 5: A*p:
 		gpu_csr_matvec_product_par<<<256,256>>>(N, gpu_csr_rowoffsets, gpu_csr_colindices, gpu_csr_values, gpu_p, gpu_Ap);	
		cudaDeviceSynchronize();
		
 
		//line 6 rhs P1
		gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm_old);
		double res_norm = 1;
		cudaMemcpy(&res_norm, gpu_res_norm_old, sizeof(double), cudaMemcpyDeviceToHost);
		double beta = res_norm; 
		cudaDeviceSynchronize();
  
		//line 6 rhs P2
		gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_p, gpu_Ap, N, gpu_pAp);
		double pAp = 0;
		cudaMemcpy(&pAp, gpu_pAp, sizeof(double), cudaMemcpyDeviceToHost); 
		cudaDeviceSynchronize();
  
		gpu_line6and7<<<256,256>>>(N, gpu_solution, gpu_res_norm_old, gpu_pAp, gpu_alpha, gpu_p);
		double alpha = 0;
		cudaMemcpy(&alpha, gpu_alpha, sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << "alpha: " << alpha << std::endl;

		cudaDeviceSynchronize();
		gpu_line8<<<256,256>>>(N, gpu_r, gpu_alpha, gpu_Ap);
		cudaDeviceSynchronize();
		 
		// line 9
		gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm_new);
		cudaMemcpy(&res_norm, gpu_res_norm_new, sizeof(double), cudaMemcpyDeviceToHost);		
 
		//line 10
		std::cout << "residual_norm_squared: " << res_norm << std::endl;
		if (res_norm < 1e-7) break;
 
		// line 11: compute beta
		beta = res_norm / beta;
		std::cout << "beta: " << beta << std::endl;
 
		// line 12: update p
		gpu_line12<<<256,256>>>(N, gpu_p, gpu_r, gpu_res_norm_old, gpu_res_norm_new);
		cudaDeviceSynchronize();
 
		if (iters > max_iters) break;  // solver didn't converge
		++iters;
 
		std::cout << "------" << std::endl;
	}
 
	cudaMemcpy(solution, gpu_solution, N * sizeof(double), cudaMemcpyDeviceToHost);
	std::cout << "Conjugate Gradients converged in " << iters << " iterations." << std::endl;
	
	cudaFree(gpu_solution);
	cudaFree(gpu_csr_rowoffsets);
	cudaFree(gpu_csr_colindices);
	cudaFree(gpu_csr_values);
	cudaFree(gpu_p);
	cudaFree(gpu_Ap);
	cudaFree(gpu_r);
	cudaFree(gpu_alpha);
	cudaFree(gpu_beta);
	cudaFree(gpu_pAp);
	cudaFree(gpu_res_norm_old);
	cudaFree(gpu_res_norm_new);	
}
 
 
 
/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction, int max_iters = 100, bool print_solution = false) {
 
	size_t N = points_per_direction * points_per_direction; // number of unknows to solve for
 
	//
	// Allocate CSR arrays.
	//
	// Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
	//       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
	//
	int *csr_rowoffsets =    (int*)malloc(sizeof(int) * (N+1));
	int *csr_colindices =    (int*)malloc(sizeof(int) * 5 * N);
	double *csr_values  = (double*)malloc(sizeof(double) * 5 * N);
 
	//
	// fill CSR matrix with values
	//
	generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);
	
			
	//
	// Allocate solution vector and right hand side:
	//
	double *solution = (double*)malloc(sizeof(double) * N);
	double *rhs      = (double*)malloc(sizeof(double) * N);
	std::fill(rhs, rhs + N, 1);
 
	//
	// Call Conjugate Gradient implementation
	//
	//conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution, max_iters);
	gpu_conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution, max_iters);
 
	//
	// Check for convergence:
	//
	double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
	std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;
	
	if (print_solution){
		printf("printing solution: \n");
		for(size_t i = 0; i < N; i++){
			printf("%1.18e\n",solution[i]);
		}
	}	
	free(solution);
	free(rhs);
	free(csr_rowoffsets);
	free(csr_colindices);
	free(csr_values);

}
 
 
int main() {
	
	int grid_size = 5;
	int max_its = 10000;
	Timer mytimer;
	
	printf("starting conj. grad for %i unknowns.\n Max. iterations: %i\n", grid_size*grid_size, max_its);

	mytimer.reset();	
	solve_system(grid_size,max_its); // solves a system with 100*100 unknowns, max_iters = 300
	
	printf("\n------------------------------\n");
	printf("%i unknowns.\n Max. iterations: %i\n", grid_size*grid_size, max_its);
	printf("elapsed time in seconds: %2.3f\n", mytimer.get());
	cudaDeviceReset();  // for CUDA leak checker to work	

	return EXIT_SUCCESS;
}