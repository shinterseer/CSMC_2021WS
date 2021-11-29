#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define GRID_SIZE 256
#define BLOCK_SIZE 256

#define ALL_MASK 0xffffffff





// _______________________ Conjugate Gradient _____________________________
// ________________________________________________________________________


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

	int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

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
		
	// rr has been initialized outside the kernel
  if (threadIdx.x == 0) atomicAdd(rr, shared_mem[0]);	
}


__global__ void cuda_cg_red(const int N, const int *csr_rowoffsets,
														const int *csr_colindices, const double *csr_values,
														const double *p, double *Ap, double *ApAp, double *pAp)
{	
	int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

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
	// ApAp is initialized outside the kernel
	if ((threadIdx.x % warpSize) == 0) atomicAdd(ApAp, ApAp_local);


	// warp reduction pAp
	for(int stride = warpSize/2; stride > 0; stride /= 2){
		__syncwarp();
		pAp_local += __shfl_down_sync(ALL_MASK, pAp_local, stride);			
	}
	__syncwarp();	
	// pAp is initialized outside the kernel
	if ((threadIdx.x % warpSize) == 0) atomicAdd(pAp, pAp_local);

}


void conjugate_gradient_pipe(const int N, // number of unknows
                        const int *csr_rowoffsets, const int *csr_colindices,
                        const double *csr_values, const double *rhs, double *solution,
												const int max_its = 10000)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

	const double zero = 0;
  // initialize work vectors:
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
	double *cuda_pAp, *cuda_rr, *cuda_ApAp;
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_ApAp, sizeof(double));

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
	
	// line 5: compute <Ap,Ap>
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
	double host_ApAp = 0;
  cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
	
	// line 5: beta = alpha^2 * ApAp / rr - 1
	double host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;

  const double initial_rr = host_rr;
  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

		//cudaDeviceSynchronize();

		cudaMemcpy(cuda_rr, &zero, sizeof(double), cudaMemcpyHostToDevice);			
		cuda_cg_blue<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
														 cuda_solution, cuda_r, cuda_p, cuda_Ap, cuda_rr,
														 host_alpha, host_beta);
		cudaDeviceSynchronize();
		cudaMemcpy(&host_rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

		// break condition
    if (std::sqrt(host_rr / initial_rr) < 1e-6) 
			break;

		
		cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);		
		cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);		
		cuda_cg_red<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets,
														csr_colindices, csr_values,
														cuda_p, cuda_Ap, cuda_ApAp, cuda_pAp);
		cudaDeviceSynchronize();
		
		cudaMemcpy(&host_ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);

		// line 13
		host_alpha = host_rr / host_pAp;

		// line 14
		host_beta = host_alpha*host_alpha * host_ApAp / host_rr - 1;
		
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
  cudaFree(cuda_pAp);
  cudaFree(cuda_rr);
  cudaFree(cuda_ApAp);	
}



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


void my_generate_system(int N, int M, int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values)
{
		//int N = 100;
		//int M = 100;
		
		int *num_entries = (int *)malloc(sizeof(int) * N*M);
		//int *csr_rowoffsets = (int *)malloc(sizeof(int) * N*M + 1);
		
		int* device_num_entries;
		cudaMalloc(&device_num_entries, sizeof(int) * N*M);
		
		int* device_csr_rowoffsets;
		cudaMalloc(&device_csr_rowoffsets, sizeof(int) * N*M + 1);	

		// get number of nonzero entries per row
		num_nonzero_entries(N, M, num_entries);
		cudaMemcpy(device_num_entries, num_entries, sizeof(int) * N*M, cudaMemcpyHostToDevice);
		
		// get row offsets for CSR as exclusive prefix sum of num_entries	
		exclusive_scan(device_num_entries, device_csr_rowoffsets, N*M);
		cudaMemcpy(csr_rowoffsets, device_csr_rowoffsets, sizeof(int) * N*M, cudaMemcpyDeviceToHost);	

		csr_rowoffsets[N*M] = csr_rowoffsets[N*M - 1] + num_entries[N*M - 1];
			
		int num_values = csr_rowoffsets[N*M];
		//int *csr_colindices = (int *)malloc(sizeof(int) * num_values);
		//double *csr_values = (double *)malloc(sizeof(double) * num_values);

		assembleA(csr_rowoffsets, csr_values, csr_colindices, N, M);		

}

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
	
  //generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
  //                     csr_values);

  my_generate_system(points_per_direction, points_per_direction, 
										 csr_rowoffsets, csr_colindices,
                     csr_values);

	/*
	//-----------------------------------------------------------------------------
	{
		int N = 100;
		int M = 100;
		
		int *num_entries = (int *)malloc(sizeof(int) * N*M);
		//int *csr_rowoffsets = (int *)malloc(sizeof(int) * N*M + 1);
		
		int* device_num_entries;
		cudaMalloc(&device_num_entries, sizeof(int) * N*M);
		
		int* device_csr_rowoffsets;
		cudaMalloc(&device_csr_rowoffsets, sizeof(int) * N*M + 1);	

		// get number of nonzero entries per row
		num_nonzero_entries(N, M, num_entries);
		cudaMemcpy(device_num_entries, num_entries, sizeof(int) * N*M, cudaMemcpyHostToDevice);
		
		// get row offsets for CSR as exclusive prefix sum of num_entries	
		exclusive_scan(device_num_entries, device_csr_rowoffsets, N*M);
		cudaMemcpy(csr_rowoffsets, device_csr_rowoffsets, sizeof(int) * N*M, cudaMemcpyDeviceToHost);	

		csr_rowoffsets[N*M] = csr_rowoffsets[N*M - 1] + num_entries[N*M - 1];
			
		int num_values = csr_rowoffsets[N*M];
		//int *csr_colindices = (int *)malloc(sizeof(int) * num_values);
		//double *csr_values = (double *)malloc(sizeof(double) * num_values);

		assembleA(csr_rowoffsets, csr_values, csr_colindices, N, M);		
	}


	//-----------------------------------------------------------------------------
	*/


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
  conjugate_gradient_pipe(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution, max_its);

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



// _________________________________ Main ________________________________
// ________________________________________________________________________

int main() {
	
	
	solve_system(100);
	/*
	
	int N = 100;
	int M = 100;
	
	bool sanity_check = false;
	
	
	int *num_entries = (int *)malloc(sizeof(int) * N*M);
	int *csr_rowoffsets = (int *)malloc(sizeof(int) * N*M + 1);
	//int *csr_rowoffsets = (int *)malloc(sizeof(int) * N*M);
	
	int* device_num_entries;
	cudaMalloc(&device_num_entries, sizeof(int) * N*M);
	
	int* device_csr_rowoffsets;
	cudaMalloc(&device_csr_rowoffsets, sizeof(int) * N*M + 1);
	

	// get number of nonzero entries per row
	num_nonzero_entries(N, M, num_entries);
	cudaMemcpy(device_num_entries, num_entries, sizeof(int) * N*M, cudaMemcpyHostToDevice);
	
	// get row offsets for CSR as exclusive prefix sum of num_entries	
	exclusive_scan(device_num_entries, device_csr_rowoffsets, N*M);
	cudaMemcpy(csr_rowoffsets, device_csr_rowoffsets, sizeof(int) * N*M, cudaMemcpyDeviceToHost);	

	csr_rowoffsets[N*M] = csr_rowoffsets[N*M - 1] + num_entries[N*M - 1];
		
	int num_values = csr_rowoffsets[N*M];
	int *csr_colindices = (int *)malloc(sizeof(int) * num_values);
	double *csr_values = (double *)malloc(sizeof(double) * num_values);

	assembleA(csr_rowoffsets, csr_values, csr_colindices, N, M);
	
	if (N !=M){
		std::cout << "N!=M aborting..." << std::endl;
		return 1;
	}
	


  generate_fdm_laplace(N, csr_rowoffsets, csr_colindices, csr_values);

	
	// move csr matrix to device

	//int* device_csr_rowoffsets;
	//cudaMalloc(&device_csr_rowoffsets, sizeof(int) * N*M + 1);
	cudaMemcpy(device_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N*M + 1), cudaMemcpyHostToDevice);
	int* device_csr_colindices;
	cudaMalloc(&device_csr_colindices, sizeof(int) * num_values);
	cudaMemcpy(device_csr_colindices, csr_colindices, sizeof(int) * num_values, cudaMemcpyHostToDevice);
	double* device_csr_values;
	cudaMalloc(&device_csr_values, sizeof(double) * num_values);
	cudaMemcpy(device_csr_values, csr_values, sizeof(double) * num_values, cudaMemcpyHostToDevice);
	

	// allocate solution and rhs 
  double *solution = (double *)malloc(sizeof(double) * N*M);
  double *rhs = (double *)malloc(sizeof(double) * N*M);
  std::fill(rhs, rhs + N*M, 1);
	
	

	conjugate_gradient_pipe(N*M, // number of unknows
                        device_csr_rowoffsets, device_csr_colindices,
                        device_csr_values, rhs, solution,
												10000);



	cudaFree(device_csr_colindices);
	cudaFree(device_csr_values);




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
*/


  return EXIT_SUCCESS;
}


