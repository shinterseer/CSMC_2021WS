#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"
#include "hip/hip_runtime.h"
 
#define GRID_SIZE 256
#define BLOCK_SIZE 256


__global__
void gpu_dotproduct_atomicAdd(const double *gpu_x, const double *gpu_y, const size_t size, double *gpu_result){
	
	size_t thread_id_global = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
	if (thread_id_global == 0)
		*gpu_result = 0;
	
	__shared__ double shared_m[BLOCK_SIZE];
		
	// I think, this is the right way:
	double thread_dp = 0;
	for (unsigned int i = thread_id_global; i<size; i += hipBlockDim_x * hipGridDim_x)
		thread_dp += gpu_x[i] * gpu_y[i];
	shared_m[hipThreadIdx_x] = thread_dp;
		
	// now the reduction
	for(int stride = hipBlockDim_x/2; stride>0; stride/=2){
		__syncthreads();
		if (hipThreadIdx_x < stride){
			shared_m[hipThreadIdx_x] += shared_m[hipThreadIdx_x + stride];
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (hipThreadIdx_x == 0){
		atomicAdd(gpu_result, shared_m[0]);		
		//gpu_result_stage1[blockIdx.x] = shared_m[0];
	}	
}


 
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
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to hip kernels.
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
	hipMalloc(&gpu_solution, N * sizeof(double));
	int *gpu_csr_rowoffsets, *gpu_csr_colindices;	
	hipMalloc(&gpu_csr_rowoffsets, (N+1) * sizeof(int));
	hipMalloc(&gpu_csr_colindices, 5*N * sizeof(int));
	double *gpu_csr_values;
	hipMalloc(&gpu_csr_values, 5*N * sizeof(double));
	// copy over to gpu
	hipMemcpy(gpu_solution, solution, N * sizeof(double), hipMemcpyHostToDevice);
	hipMemcpy(gpu_csr_rowoffsets, csr_rowoffsets, (N+1) * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(gpu_csr_colindices, csr_colindices, 5*N * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(gpu_csr_values, csr_values, 5*N * sizeof(double), hipMemcpyHostToDevice);
 
	
	
	// initialize work vectors:
	const double *zero = 0;
	double *r = (double*)malloc(sizeof(double) * N);
	double *p = (double*)malloc(sizeof(double) * N);
	double *Ap = (double*)malloc(sizeof(double) * N);
 
	double *gpu_p, *gpu_Ap;
	hipMalloc(&gpu_p, N * sizeof(double));
	hipMalloc(&gpu_Ap, N * sizeof(double));
	double *gpu_r, *gpu_alpha, *gpu_beta, *gpu_pAp;
	hipMalloc(&gpu_r, N * sizeof(double));
	//hipMalloc(&gpu_res_norm, sizeof(double));
	hipMalloc(&gpu_alpha, sizeof(double));
	hipMalloc(&gpu_beta, sizeof(double));
	hipMalloc(&gpu_pAp, sizeof(double));
	double *gpu_res_norm_old, *gpu_res_norm_new;
	hipMalloc(&gpu_res_norm_old, sizeof(double));
	hipMalloc(&gpu_res_norm_new, sizeof(double));
	
	// line 2: initialize r and p:
	std::copy(rhs, rhs+N, p);
	std::copy(rhs, rhs+N, r);
	hipMemcpy(gpu_p, p, sizeof(double) * N, hipMemcpyHostToDevice);		
	hipMemcpy(gpu_r, r, sizeof(double) * N, hipMemcpyHostToDevice);
 
	int iters = 0;
	while (1) {
 
		// line 5: A*p:
 		// gpu_csr_matvec_product_par<<<256,256>>>(N, gpu_csr_rowoffsets, gpu_csr_colindices, gpu_csr_values, gpu_p, gpu_Ap);	
		hipLaunchKernelGGL(gpu_csr_matvec_product_par,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 N, gpu_csr_rowoffsets, gpu_csr_colindices, gpu_csr_values, gpu_p, gpu_Ap);
		hipDeviceSynchronize();
		
 
		//line 6 rhs P1
		
		// hipMemcpy(gpu_res_norm_old, zero, sizeof(double), hipMemcpyHostToDevice);				
		// gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm_old);
		hipLaunchKernelGGL(gpu_dotproduct_atomicAdd,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 gpu_r, gpu_r, N, gpu_res_norm_old);
		double res_norm = 1;
		hipMemcpy(&res_norm, gpu_res_norm_old, sizeof(double), hipMemcpyDeviceToHost);
		double beta = res_norm; 
		hipDeviceSynchronize();
  
		//line 6 rhs P2
		// hipMemcpy(gpu_pAp, zero, sizeof(double), hipMemcpyHostToDevice);				
		// gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_p, gpu_Ap, N, gpu_pAp);
		hipLaunchKernelGGL(gpu_dotproduct_atomicAdd,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 gpu_p, gpu_Ap, N, gpu_pAp);
		double pAp = 0;
		hipMemcpy(&pAp, gpu_pAp, sizeof(double), hipMemcpyDeviceToHost); 
		hipDeviceSynchronize();
  
		// gpu_line6and7<<<256,256>>>(N, gpu_solution, gpu_res_norm_old, gpu_pAp, gpu_alpha, gpu_p);
		hipLaunchKernelGGL(gpu_line6and7,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 N, gpu_solution, gpu_res_norm_old, gpu_pAp, gpu_alpha, gpu_p);
		double alpha = 0;
		hipMemcpy(&alpha, gpu_alpha, sizeof(double), hipMemcpyDeviceToHost);
		std::cout << "alpha: " << alpha << std::endl;

		hipDeviceSynchronize();
		// gpu_line8<<<256,256>>>(N, gpu_r, gpu_alpha, gpu_Ap);
		hipLaunchKernelGGL(gpu_line8,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 N, gpu_r, gpu_alpha, gpu_Ap);
		hipDeviceSynchronize();
		 
		// line 9
		hipMemcpy(gpu_res_norm_new, zero, sizeof(double), hipMemcpyHostToDevice);						
		// gpu_dotproduct_atomicAdd<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm_new);
		hipLaunchKernelGGL(gpu_dotproduct_atomicAdd,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 gpu_r, gpu_r, N, gpu_res_norm_new);
		hipMemcpy(&res_norm, gpu_res_norm_new, sizeof(double), hipMemcpyDeviceToHost);		
 
		//line 10
		std::cout << "residual_norm_squared: " << res_norm << std::endl;
		if (res_norm < 1e-7) break;
 
		// line 11: compute beta
		beta = res_norm / beta;
		std::cout << "beta: " << beta << std::endl;
 
		// line 12: update p
		// gpu_line12<<<256,256>>>(N, gpu_p, gpu_r, gpu_res_norm_old, gpu_res_norm_new);
		hipLaunchKernelGGL(gpu_line12,
											 dim3(GRID_SIZE), dim3(BLOCK_SIZE),
											 0,0,
											 N, gpu_p, gpu_r, gpu_res_norm_old, gpu_res_norm_new);
		hipDeviceSynchronize();
 
		if (iters > max_iters) break;  // solver didn't converge
		++iters;
 
		std::cout << "------" << std::endl;
	}
 
	hipMemcpy(solution, gpu_solution, N * sizeof(double), hipMemcpyDeviceToHost);
	std::cout << "Conjugate Gradients converged in " << iters << " iterations." << std::endl;
	
	hipFree(gpu_solution);
	hipFree(gpu_csr_rowoffsets);
	hipFree(gpu_csr_colindices);
	hipFree(gpu_csr_values);
	hipFree(gpu_p);
	hipFree(gpu_Ap);
	hipFree(gpu_r);
	hipFree(gpu_alpha);
	hipFree(gpu_beta);
	hipFree(gpu_pAp);
	hipFree(gpu_res_norm_old);
	hipFree(gpu_res_norm_new);	
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
	hipDeviceReset();  // for hip leak checker to work	

	return EXIT_SUCCESS;
}