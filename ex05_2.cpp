#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define GRID_SIZE 512
#define BLOCK_SIZE 512


// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }
    y[i] = sum;
  }
}

// cuda_vecadd: x <- x + alpha * y
__global__ void cuda_vecadd(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] += alpha * y[i];
}


// cuda_vecadd2: x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] = y[i] + alpha * x[i];
}

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
	int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	
	if (global_thread_idx == 0)
		*result = 0;
	
  __shared__ double shared_mem[BLOCK_SIZE];

  double dot = 0;
  for (int i = global_thread_idx; i < N; i += num_threads) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}


void conjugate_gradient_pipe(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_scalar;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_scalar, sizeof(double));
	double *cuda_alpha, *cuda_beta, *cuda_pAp, *cuda_rr, *cuda_ApAp;
  cudaMalloc(&cuda_alpha, sizeof(double));
  cudaMalloc(&cuda_beta, sizeof(double));
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_ApAp, sizeof(double));

	// line2: x_0 = 0, Ax_0 = 0
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
	// line 2: p_0 = b
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice); 
	// line 2: r_0 = b
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);

	// line 4: compute Ap
	cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
	// line 4: compute <p,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
	// line 4: compute <r,r>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);

	// line 4
  double host_rr = 0;
  cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

	double host_pAp = 0;
  cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);

	// line 4 final: alhpa = rr / pAp
  cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
	double host_alpha = host_rr / host pAp;
  cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);
	
	// line 5: compute <Ap,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
	double host_ApAp = 0;
  cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
	
	// line 5: beta = alpha^2 * ApAp / rr - 1
	double host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
  cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

  double initial_rr = host_rr;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

		// line 7
		cuda_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_solution, cuda_p, cuda_alpha);

		// line 8
		cuda_vecsub<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_solution, cuda_p, cuda_alpha);

    // line 9:
    cuda_vecadd2<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_r, host_beta);

    // line 10:
    cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);

		// line 11 p1: compute <Ap,Ap>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
		cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 11 p2: compute <p,Ap>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
		cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 12: compute <r,r>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
		cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

		// break condition
    if (std::sqrt(host_rr / initial_rr) < 1e-6) 
			break;

		// line 13
		host_alpha = host_rr / host pAp;
	  cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);

		// line 14
		host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
		cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
	
	
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;

  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_scalar);
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {

  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);

  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;
  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
                       csr_values);

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  //
  // Allocate CUDA-arrays //
  //
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  conjugate_gradient_pipe(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;

  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}

int main() {

  solve_system(1000); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}

