	#include <stdio.h>
	#include <vector>
	#include "timer.hpp"
	#include <algorithm>

	// unused here
	__global__
	void gpu_sum(const double *gpu_x, const double *gpu_y, double* gpu_result, 
				 const size_t size){
					 
		size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		size_t num_threads = gridDim.x*blockDim.x;
		
		for(size_t i=thread_id; i<size; i += num_threads){		
			gpu_result[i] = gpu_x[i] + gpu_y[i];
		}	
	}

	// ########################################################################
	// ################## The Kernel ##########################################
	// ########################################################################
	__global__
	void gpu_sum_skip_k(const double *gpu_x, const double *gpu_y, double* gpu_result, 
						const size_t size, const size_t k){
		size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		size_t num_threads = gridDim.x*blockDim.x;
		
		for(size_t i=thread_id; i < size-k; i += num_threads){
			gpu_result[i+k] = gpu_x[i+k] + gpu_y[i+k];
		}	
	}


	void cpu_init(double *x, double *y, size_t size){
		
		for(size_t i=0; i<size; i++){
			x[i] = i;
			y[i] = size - i - 1;
		}
	}

	// for debugging
	void show_result(double* result, size_t start=3, size_t end=8){
		for(size_t i = start; i<=end; i++){
			printf("result[%zu] = %1.3e\n", i, result[i]);
		}		
	}


	// ########################################################################
	// ################## The very cool execution wrapper #####################
	// ########################################################################

	template<typename KERNEL, typename ...ARGS>
	double execution_wrapper(int grid_size, int block_size, int repetitions,
							 KERNEL gpu_kernel, ARGS... pass_this_to_kernel){

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
	// ########################### Main #######################################
	// ########################################################################

	int main(){
		
		Timer global_Timer;		
		global_Timer.reset();

		size_t size_vec = size_t(1e8);
		
		int repetitions = 10;
		int grid_size = 256;
		int block_size = 256;

		// allocate on host and device
		// ----------------------------
		double *x = (double*)malloc(size_vec*sizeof(double)); 
		double *y = (double*)malloc(size_vec*sizeof(double)); 
		double *result = (double*)malloc(size_vec*sizeof(double)); 
		
		double *gpu_x, *gpu_y, *gpu_result;
		cudaMalloc(&gpu_x, size_vec*sizeof(double));
		cudaMalloc(&gpu_y, size_vec*sizeof(double));
		cudaMalloc(&gpu_result, size_vec*sizeof(double));
		
		// initialize and copy to device
		// ------------------------------
		cpu_init(x,y,size_vec);		
		cudaMemcpy(gpu_x, x, size_vec*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_y, y, size_vec*sizeof(double), cudaMemcpyHostToDevice);	

		printf("\n");
		printf("summing vectors on GPU (skip k entries) \n");
		printf("----------------------------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("k; median elapsed time in seconds;\n");
				
		size_t k_max = 64;
		double elapsed_time = 0;
		for(size_t k = 0; k < k_max; k++){
			
			elapsed_time = execution_wrapper(grid_size, block_size, repetitions, 
											 gpu_sum_skip_k,
											 gpu_x, gpu_y, gpu_result, 
											 size_vec, k);

			cudaMemcpy(result,gpu_result,size_vec*sizeof(double),cudaMemcpyDeviceToHost);
			
			printf("%zu;", k);
			printf("%1.3e\n", elapsed_time);

		}
		cudaFree(gpu_x);
		cudaFree(gpu_y);
		cudaFree(gpu_result);
		free(x);
		free(y);
		free(result);

		printf("\n total elapsed time in seconds: %2.2f\n", global_Timer.get());		
		return 0;		
	}
