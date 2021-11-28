
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"



__global__
void gpu_dotp_sequ(const double *x, const double *y, const size_t size, double *dotp){
	
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;

	if (thread_id_global == 0){
		*dotp = 0;
		
		for(size_t i = 0; i < size; i++){
			*dotp += x[i] * y[i];
		}		
	}
}


__global__
void gpu_dotp_wshuffle(const double *x, const double *y, const size_t size, double *dotp){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	// sum of entries
	double thread_dotp = 0;
	
	if (thread_id_global == 0){
		*dotp = 0;
	}

	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_dotp += x[i] * y[i];
	}

	
	// now the reduction inside the warp
	int shuffle_delta;
	int lane = threadIdx.x % warpSize;
	for(int stride = 1; stride <= warpSize/2; stride *= 2){
		
		// if you are lower half -> get value from upper half and vice versa
		if ((lane % (2*stride)) < stride)
			shuffle_delta = stride;
		else
			shuffle_delta = (-1)*stride;
		
		__syncwarp();
		thread_dotp += __shfl_down_sync(-1, thread_dotp, shuffle_delta);
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(dotp, thread_dotp);
	}	
}

__global__
void gpu_line6and7(const double* res_norm, double *alpha, const double *p, const size_t size, double *x){
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	*alpha = *res_norm / *alpha;
	if (thread_id_global == 0) printf("alpha: %3.10f\n",*alpha);
	for (size_t i = thread_id_global; i < size; i += thread_num){
		x[i] += (*alpha) * p[i];
	}
}


__global__
void gpu_line7(const double *alpha, const double *p, const size_t size, double *x){
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	for (size_t i = thread_id_global; i < size; i += thread_num){
		x[i] += (*alpha) * p[i];		
	}
}


__global__
void gpu_line8(const double *scalar, const double *vector, const size_t size, double *result){	
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	for (size_t i = thread_id_global; i < size; i += thread_num){
		result[i] -= (*scalar) * vector[i];		
	}
}

__global__
void gpu_line12(double *p, const double *r, const double *beta, const size_t size){		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	for (size_t i = thread_id_global; i < size; i += thread_num){
		p[i] = r[i] + (*beta) * p[i];		
	}
}


/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
__global__
void gpu_csr_matvec_product(const size_t N,
                            const int *csr_rowoffsets, const int *csr_colindices, const double *csr_values,
                            const double *x, double *y)
{
  /** YOUR CODE HERE
   *
   *  Either provide a CPU implementation, or call the CUDA kernel here
   *
   */

	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread_id_global == 0){
		for (size_t row=0; row < N; ++row) {
			double val = 0; // y = Ax for this row
			for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row+1]; ++jj) {
				val += csr_values[jj] * x[csr_colindices[jj]];
			}
			y[row] = val;
		}	
	}
}

__global__
void gpu_csr_matvec_product_par(const size_t N,
                            const int *csr_rowoffsets, const int *csr_colindices, const double *csr_values,
                            const double *x, double *y)
{
  /** YOUR CODE HERE
   *
   *  Either provide a CPU implementation, or call the CUDA kernel here
   *
   */

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





/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *x, double *y)
{
  /** YOUR CODE HERE
   *
   *  Either provide a CPU implementation, or call the CUDA kernel here
   *
   */


  for (size_t row=0; row < N; ++row) {
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
	double *gpu_r, *gpu_res_norm, *gpu_alpha, *gpu_beta;
	cudaMalloc(&gpu_r, N * sizeof(double));
	cudaMalloc(&gpu_res_norm, sizeof(double));
	cudaMalloc(&gpu_alpha, sizeof(double));
	cudaMalloc(&gpu_beta, sizeof(double));
	
  // line 2: initialize r and p:
  std::copy(rhs, rhs+N, p);
  std::copy(rhs, rhs+N, r);
	cudaMemcpy(gpu_p, p, sizeof(double) * N, cudaMemcpyHostToDevice);		
	cudaMemcpy(gpu_r, r, sizeof(double) * N, cudaMemcpyHostToDevice);

  int iters = 0;
  while (1) {

    // line 5: A*p:
    //csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);
		cudaMemcpy(gpu_p, p, sizeof(double) * N, cudaMemcpyHostToDevice);
		//cudaMemcpy(gpu_Ap, Ap, sizeof(double) * N, cudaMemcpyHostToDevice);

    //gpu_csr_matvec_product<<<1,1>>>(N, gpu_csr_rowoffsets, gpu_csr_colindices, gpu_csr_values, gpu_p, gpu_Ap);
    gpu_csr_matvec_product_par<<<256,256>>>(N, gpu_csr_rowoffsets, gpu_csr_colindices, gpu_csr_values, gpu_p, gpu_Ap);
		
		//cudaMemcpy(p, gpu_p, sizeof(double) * N, cudaMemcpyDeviceToHost);
		//cudaMemcpy(Ap, gpu_Ap, sizeof(double) * N, cudaMemcpyDeviceToHost); // this also provides impicit sync
		cudaDeviceSynchronize();
		
    // similarly for the other operations

    // line 6:
    double res_norm = 0;
    //for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];

		//line 6 P1
		//cudaMemcpy(gpu_r, r, sizeof(double) * N, cudaMemcpyHostToDevice);		
		gpu_dotp_wshuffle<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm);		
		cudaMemcpy(&res_norm, gpu_res_norm, sizeof(double), cudaMemcpyDeviceToHost);		
		//cudaDeviceSynchronize();


    double alpha = 0;		
    //for (size_t i=0; i<N; ++i) alpha += p[i] * Ap[i];

		//line 6 P2
		gpu_dotp_wshuffle<<<256,256>>>(gpu_p, gpu_Ap, N, gpu_alpha);		
		cudaMemcpy(&alpha, gpu_alpha, sizeof(double), cudaMemcpyDeviceToHost);		
		
    alpha = res_norm / alpha;
    //std::cout << "alpha: " << alpha << std::endl;

		cudaMemcpy(gpu_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
		//cudaDeviceSynchronize();

    // line 7,8:
    //for (size_t i=0; i<N; ++i) {
    //  solution[i] += alpha *  p[i];
    //  r[i]        -= alpha * Ap[i];
    //}
		//gpu_line6and7<<<256,256>>>(gpu_res_norm, gpu_alpha, gpu_p, N, gpu_solution);
		
		gpu_line7<<<256,256>>>(gpu_alpha, gpu_p, N, gpu_solution);
		//cudaDeviceSynchronize();
		gpu_line8<<<256,256>>>(gpu_alpha, gpu_Ap, N, gpu_r);
		cudaMemcpy(r, gpu_r, N * sizeof(double), cudaMemcpyDeviceToHost);

    double beta = res_norm;

    // lines 9, 10:
    //res_norm = 0;
    //for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];
		gpu_dotp_wshuffle<<<256,256>>>(gpu_r, gpu_r, N, gpu_res_norm);
		cudaMemcpy(&res_norm, gpu_res_norm, sizeof(double), cudaMemcpyDeviceToHost);		
		
    std::cout << "residual_norm_squared: " << res_norm << std::endl;
    if (res_norm < 1e-7) break;

    // line 11: compute beta
    beta = res_norm / beta;
    std::cout << "beta: " << beta << std::endl;

    // line 12: update p
    //for (size_t i=0; i<N; ++i) p[i] = r[i] + beta * p[i];
		cudaMemcpy(gpu_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
		gpu_line12<<<256,256>>>(gpu_p, gpu_r, gpu_beta, N);
		cudaMemcpy(p, gpu_p, N * sizeof(double), cudaMemcpyDeviceToHost);

    if (iters > max_iters) break;  // solver didn't converge
    ++iters;

    std::cout << "------" << std::endl;
  }

	cudaMemcpy(solution, gpu_solution, N * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "Conjugate Gradients converged in " << iters << " iterations." << std::endl;
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
	
}


int main() {
	
	int grid_size = 1000;
	int max_its = 10000;

	printf("starting conj. grad for %i unknowns.\n Max. iterations: %i\n", grid_size*grid_size, max_its);

  solve_system(grid_size,max_its); // solves a system with 100*100 unknowns, max_iters = 300

  return EXIT_SUCCESS;
}