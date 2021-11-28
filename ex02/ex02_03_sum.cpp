	#include <stdio.h>
	//#include "omp.h"
	#include <vector>
	#include "timer.hpp"


	void cpuwork(int amount){
		volatile int blub = 0;
		for(int i=0; i<amount;i++){
			blub++;
			blub--;
		}
	}

	__global__ 
	void gpu_init(double *gpu_x, double *gpu_y, size_t size){

		size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		for(size_t i=thread_id; i<size; i += blockDim.x*gridDim.x){
		//for(int i=0; i<size; i++){
			gpu_x[i] = i;
			gpu_y[i] = size - i - 1;
		}
	}

	__global__
	void gpu_sum(const double *gpu_x, const double *gpu_y, double* gpu_result, const size_t size){
		size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		for(size_t i=thread_id; i<size; i += blockDim.x*gridDim.x){		
			gpu_result[i] = gpu_x[i] + gpu_y[i];
		}	
	}


	void cpu_init(double *x, double *y, size_t size){
		
		for(size_t i=0; i<size; i++){
			x[i] = i;
			y[i] = size - i - 1;
		}
	}

	void show_result(double* result, size_t start=3, size_t end=8){
		for(size_t i = start; i<=end; i++){
			printf("result[%zu] = %1.3e\n", i, result[i]);
		}		
	}

	int main(){
		
		Timer mytimer, global_Timer;
		double elapsed_time;
		
		global_Timer.reset();

		/*
		std::vector<size_t> sizes;
		int numsizes = 14;
		size_t tmp_size = 100;

		for(int i=0;i<numsizes; i+=2){
			sizes.push_back(tmp_size);
			sizes.push_back(3*tmp_size);
			tmp_size *= 10;
		}
		*/
		
		int numsizes = 1;
		std::vector<size_t> sizes;
		sizes.push_back(size_t(1e7));
		
		int repetitions;

		repetitions = 1;
		printf("\n");
		printf("summing vectors on GPU \n");
		printf("------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds;\n");
		for(int i=0; i<int(sizes.size()); i++){
			
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double)); 
			double *result = (double*)malloc(sizes[i]*sizeof(double)); 
			
			double *gpu_x, *gpu_y, *gpu_result;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_result, sizes[i]*sizeof(double));
			
			cpu_init(x,y,sizes[i]);
			
			cudaMemcpy(gpu_x,x,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y,y,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);	

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
									
			//for(int j=0; j<repetitions; j++){
				gpu_sum<<<256,256>>>(gpu_x,gpu_y,gpu_result,sizes[i]);
			//}
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();

			cudaMemcpy(result,gpu_result,sizes[i]*sizeof(double),cudaMemcpyDeviceToHost);

			//show_result(result);
			
			printf("%zu;", sizes[i]);
			printf("%1.3e\n", elapsed_time);

			cudaFree(gpu_x);
			cudaFree(gpu_y);
			cudaFree(gpu_result);
			free(x);
			free(y);
			free(result);

		}

		printf("\n total elapsed time in seconds: %1.3e\n", global_Timer.get());		
		return 0;		
	}
