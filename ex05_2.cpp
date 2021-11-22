#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

//#define GRID_SIZE 512
//#define BLOCK_SIZE 512
#define GRID_SIZE 32
#define BLOCK_SIZE 32

#define ALL_MASK 0xffffffff


// y = A * x
__global__ void cuda_csr_matvec_product(const int N, const int *csr_rowoffsets,
                                        const int *csr_colindices, const double *csr_values,
                                        const double *x, double *y)
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
__global__ void cuda_vecadd(const int N, double *x, const double *y, const double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] += alpha * y[i];
}


// cuda_vecadd2: x <- y + alpha * x
__global__ void cuda_vecadd2(const int N, double *x, const double *y, const double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] = y[i] + alpha * x[i];
}

// result = (x, y)
__global__ void cuda_dot_product(const int N, const double *x, const double *y, double *result)
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

__global__ void cuda_cg_blue(const int N, 
												 double *x, double *r, double *p, const double *Ap, double *rr, 
												 const double alpha, const double beta){
//__global__ void cuda_cg_blue(const int N, 
//												 double *x, double *r, double *p, const double * Ap, double *cuda_rr, 
//												 const double *alpha, const double *beta){


	int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	// temp... this is not safe here
	if (global_thread_idx == 0) *rr = 0;

	double p_old, r_new;
	double rr_local = 0;
  for (size_t i = global_thread_idx; i < N; i += num_threads){
		p_old = p[i];
		
		// line 7
    x[i] += alpha * p_old;
	
		// line 8
		r_new = r[i] - alpha * Ap[i]; 
		r[i] = r_new;
		
		// line 9
		p[i] = r_new + beta*p_old;

		// <r,r>
		rr_local += r_new * r_new;
	}
	
	// reduction for scalar product <r,r>
	__shared__ double shared_mem[BLOCK_SIZE];
  shared_mem[threadIdx.x] = rr_local;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }
		
	// initializing *cuda_rr = 0 here is too late (no global thread sync is possible)
	// therefore *cuda_rr = 0 happens in the red kernel instead
  if (threadIdx.x == 0) atomicAdd(rr, shared_mem[0]);
	
}


__global__ void cuda_cg_red(const int N, const int *csr_rowoffsets,
														const int *csr_colindices, const double *csr_values,
														const double *p, double *Ap, double *ApAp, double *pAp)
{
	
	int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	// temp... this is not safe here
	if (global_thread_idx == 0) {
		*pAp = 0;
		*ApAp = 0;
	}
	
	double Ap_local;
	double ApAp_local = 0;
	double pAp_local = 0;
  for (int i = global_thread_idx; i < N; i += num_threads) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * p[csr_colindices[k]];
    }
		Ap_local = sum;
		
		// Ap
    Ap[i] = Ap_local;
		
		// <p,Ap>		
		pAp_local += p[i] * Ap_local;
		// <Ap,Ap>				
		ApAp_local += Ap_local * Ap_local;
  }
	
	
	// warp reduction ApAp
	for(int stride = warpSize/2; stride > 0; stride /= 2){
		__syncwarp();
		ApAp_local += __shfl_down_sync(ALL_MASK, ApAp_local, stride);			
	}
	__syncwarp();	
	if ((threadIdx.x % warpSize) == 0) atomicAdd(ApAp, ApAp_local);


	// warp reduction pAp
	for(int stride = warpSize/2; stride > 0; stride /= 2){
		__syncwarp();
		pAp_local += __shfl_down_sync(ALL_MASK, pAp_local, stride);			
	}
	__syncwarp();	
	if ((threadIdx.x % warpSize) == 0) atomicAdd(pAp, pAp_local);


	
	
	/*
	// reduction for scalar product <Ap,Ap>
	__shared__ double shared_mem_ApAp[BLOCK_SIZE];
  shared_mem_ApAp[threadIdx.x] = pAp_local;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem_ApAp[threadIdx.x] += shared_mem_ApAp[threadIdx.x + k];
    }
  }
		
	// initializing *ApPp = 0 here is too late (no global thread sync is possible)
	// therefore *ApAp = 0 happens in the red kernel instead
  if (threadIdx.x == 0) atomicAdd(ApAp, shared_mem_ApAp[0]);
	*/
  //atomicAdd(ApAp, ApAp_local);

	/*
	// reduction for scalar product <p,Ap>
	__shared__ double shared_mem_pAp[BLOCK_SIZE];
  shared_mem_pAp[threadIdx.x] = pAp_local;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem_pAp[threadIdx.x] += shared_mem_pAp[threadIdx.x + k];
    }
  }
		
	// initializing *pPp = 0 here is too late (no global thread sync is possible)
	// therefore *pAp = 0 happens in the red kernel instead
  if (threadIdx.x == 0) atomicAdd(pAp, shared_mem_pAp[0]);
	*/
	
	 //atomicAdd(pAp, pAp_local);

}




void conjugate_gradient_pipe(const int N, // number of unknows
                        const int *csr_rowoffsets, const int *csr_colindices,
                        const double *csr_values, const double *rhs, double *solution,
												const int max_its = 10000)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;


  // initialize work vectors:
  //double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
	//double *cuda_scalar;
  //cudaMalloc(&cuda_scalar, sizeof(double));
	double *cuda_pAp, *cuda_rr, *cuda_ApAp;
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_ApAp, sizeof(double));
	//double *cuda_alpha, *cuda_beta;
  //cudaMalloc(&cuda_alpha, sizeof(double));
  //cudaMalloc(&cuda_beta, sizeof(double));


  // line 1: choose x_0
  std::fill(solution, solution + N, 0);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
	
	// line 2: p_0 = b
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice); 
	// line 2: r_0 = b
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);

	// line 3: compute Ap
	cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
  cudaDeviceSynchronize();

	// line 4: compute <p,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
  //cudaDeviceSynchronize();
	double host_pAp = 0;
  cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);	
	// line 4: compute <r,r>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
  //cudaDeviceSynchronize();
  double host_rr = 0;
  cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
	// line 4 final: alhpa = rr / pAp
	double host_alpha = host_rr / host_pAp;
  //cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);
	
	// line 5: compute <Ap,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
  //cudaDeviceSynchronize();
	double host_ApAp = 0;
  cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
	
	// line 5: beta = alpha^2 * ApAp / rr - 1
	double host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
  //cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

  const double initial_rr = host_rr;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

		// line 7
		cuda_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_solution, cuda_p, host_alpha);
		cudaDeviceSynchronize();

		// line 8
		cuda_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_Ap, (-1)*host_alpha);
		cudaDeviceSynchronize();

    // line 9:
    cuda_vecadd2<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_r, host_beta);
		cudaDeviceSynchronize();

    // line 10:
    cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
		cudaDeviceSynchronize();

		// line 11 p1: compute <Ap,Ap>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
		cudaDeviceSynchronize();
		cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 11 p2: compute <p,Ap>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
		cudaDeviceSynchronize();
		cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 12: compute <r,r>
		cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
		cudaDeviceSynchronize();
		cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

		// break condition
    if (std::sqrt(host_rr / initial_rr) < 1e-6) 
			break;

		// line 13
		host_alpha = host_rr / host_pAp;
	  //cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);

		// line 14
		host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
		//cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

    if (iters > max_its)
      break; // solver didn't converge
    ++iters;
  }
	
	
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  if (iters > max_its)
    std::cout << "Conjugate Gradient did NOT converge within " << max_its << " iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;

  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  //cudaFree(cuda_scalar);
}


void conjugate_gradient_pipe2(const int N, // number of unknows
                        const int *csr_rowoffsets, const int *csr_colindices,
                        const double *csr_values, const double *rhs, double *solution,
												const int max_its = 10000)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

	const double zero = 0;
  // initialize work vectors:
  //double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
	//double *cuda_scalar;
  //cudaMalloc(&cuda_scalar, sizeof(double));
	double *cuda_pAp, *cuda_rr, *cuda_ApAp;
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_ApAp, sizeof(double));
	//double *cuda_alpha, *cuda_beta;
  //cudaMalloc(&cuda_alpha, sizeof(double));
  //cudaMalloc(&cuda_beta, sizeof(double));


  // line 1: choose x_0
  std::fill(solution, solution + N, 0);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
	
	// line 2: p_0 = b
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice); 
	// line 2: r_0 = b
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);

	// line 3: compute Ap
	cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
  cudaDeviceSynchronize();

	// line 4: compute <p,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
  //cudaDeviceSynchronize();
	double host_pAp = 0;
  cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);	
	// line 4: compute <r,r>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
  //cudaDeviceSynchronize();
  double host_rr = 0;
  cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
	// line 4 final: alhpa = rr / pAp
	double host_alpha = host_rr / host_pAp;
  //cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);
	
	// line 5: compute <Ap,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
  //cudaDeviceSynchronize();
	double host_ApAp = 0;
  cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
	
	// line 5: beta = alpha^2 * ApAp / rr - 1
	double host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
  //cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

  const double initial_rr = host_rr;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

		// line 7
		//cuda_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_solution, cuda_p, host_alpha);
		//cudaDeviceSynchronize();

		// line 8
		//cuda_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_Ap, (-1)*host_alpha);
		//cudaDeviceSynchronize();

    // line 9:
    //cuda_vecadd2<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_r, host_beta);
		//cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		//cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);
		
		cuda_cg_blue<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
														 cuda_solution, cuda_r, cuda_p, cuda_Ap, cuda_rr,
														 host_alpha, host_beta);
		cudaDeviceSynchronize();
		
		
		//cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);		
		//cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);		
		cuda_cg_red<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets,
														csr_colindices, csr_values,
														cuda_p, cuda_Ap, cuda_ApAp, cuda_pAp);
		cudaDeviceSynchronize();

		
		
    // line 10:
    //cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
		//cudaDeviceSynchronize();

		// line 11 p1: compute <Ap,Ap>
		//cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
		//cudaDeviceSynchronize();
		cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 11 p2: compute <p,Ap>
		//cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
		//cudaDeviceSynchronize();
		cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 12: compute <r,r>
		//cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
		cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);		
		//cudaDeviceSynchronize();

		// break condition
    if (std::sqrt(host_rr / initial_rr) < 1e-6) 
			break;

		// line 13
		host_alpha = host_rr / host_pAp;
	  //cudaMemcpy(cuda_alpha, &host_alpha, sizeof(double), cudaMemcpyHostToDevice);

		// line 14
		host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
		//cudaMemcpy(cuda_beta, &host_beta, sizeof(double), cudaMemcpyHostToDevice);

    if (iters > max_its)
      break; // solver didn't converge
    ++iters;
  }
	
	
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  if (iters > max_its)
    std::cout << "Conjugate Gradient did NOT converge within " << max_its << " iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;

  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  //cudaFree(cuda_scalar);
}




/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction, const int max_its = 10000) {

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
  //conjugate_gradient_pipe(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution, max_its);
  conjugate_gradient_pipe2(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution, max_its);

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

	int p_grid = 100;
  solve_system(p_grid); // solves a system with p_grid*p_grid unknowns

  return EXIT_SUCCESS;
}

