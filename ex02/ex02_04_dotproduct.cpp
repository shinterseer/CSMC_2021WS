#include <stdio.h>
//#include "omp.h"
#include <vector>
#include "timer.hpp"
//#define DEBUG
#define GRID_SIZE 256
#define BLOCK_SIZE 128


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

double cpu_sum(const double *arr, size_t size){
	double sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += arr[i];
	return sum;
}

void show_result(double* result, size_t start=3, size_t end=8){
	for(size_t i = start; i <= end; i++){
		printf("result[%zu] = %1.3e\n", i, result[i]);
	}		
}


__global__
void gpu_dotproduct_stage1(const double *gpu_x, const double *gpu_y, size_t size, double *gpu_result_stage1){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ double shared_m[BLOCK_SIZE];
	
	#ifdef DEBUG
		if (thread_id_global == 0){
			printf("hello from thread 0\n");
			printf("gridDim.x = %i\n", gridDim.x);
			printf("blockDim.x = %i\n", blockDim.x);
		}
		if (thread_id_global == 25*129){
			printf("hello from thread 25*128\n");
			printf("blockIdx.x = %i\n", blockIdx.x);
			printf("threadIdx.x = %i\n", threadIdx.x);
			
		}
	#endif
	
	// lecture sais, this is correct
	/*double thread_dp = 0;
	for (unsigned int i = threadIdx.x; i<size; i += blockDim.x)
		thread_dp += gpu_x[i] * gpu_y[i];
	*/
	
	
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
	
	// thread 0 writes result
	if (threadIdx.x == 0){
		gpu_result_stage1[blockIdx.x] = shared_m[0];
	}	
}


__global__
void gpu_dotproduct_stage2(double *gpu_result_stage1, double *gpu_result_stage2){
	
	// only one block has a job here
	if (blockIdx.x == 0){
		//size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
		__shared__ double shared_m[BLOCK_SIZE];
		
		#ifdef DEBUG
			if (thread_id_global == 0){
				printf("hello from thread 0\n");
				printf("gridDim.x = %i\n", gridDim.x);
				printf("blockDim.x = %i\n", blockDim.x);
			}
			if (thread_id_global == 25*129){
				printf("hello from thread 25*128\n");
				printf("blockIdx.x = %i\n", blockIdx.x);
				printf("threadIdx.x = %i\n", threadIdx.x);
				
			}
		#endif
		
		// this time, the lecture is correct
		double thread_sum = 0;
		for (unsigned int i = threadIdx.x; i<GRID_SIZE; i += blockDim.x)
			thread_sum += gpu_result_stage1[i];
		shared_m[threadIdx.x] = thread_sum;
			
		// now the reduction
		for(int stride = blockDim.x/2; stride>0; stride/=2){
			__syncthreads();
			if (threadIdx.x < stride){
				shared_m[threadIdx.x] += shared_m[threadIdx.x + stride];
			}
		}
		//__syncthreads();
		// thread 0 writes result
		if (threadIdx.x == 0){
			//printf("hi, im thread 0 and im now writing %1.5e\n", shared_m[0]);
			*gpu_result_stage2 = shared_m[0];
		}			
	}
}


__global__
void gpu_dotproduct_atomicAdd(const double *gpu_x, const double *gpu_y, size_t size, double *gpu_result){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread_id_global == 0)
		*gpu_result = 0;
	
	__shared__ double shared_m[BLOCK_SIZE];
	
	#ifdef DEBUG
		if (thread_id_global == 0){
			printf("hello from thread 0\n");
			printf("gridDim.x = %i\n", gridDim.x);
			printf("blockDim.x = %i\n", blockDim.x);
		}
		if (thread_id_global == 25*129){
			printf("hello from thread 25*128\n");
			printf("blockIdx.x = %i\n", blockIdx.x);
			printf("threadIdx.x = %i\n", threadIdx.x);
			
		}
	#endif
	
	// lecture sais, this is correct
	/*double thread_dp = 0;
	for (unsigned int i = threadIdx.x; i<size; i += blockDim.x)
		thread_dp += gpu_x[i] * gpu_y[i];
	*/
	
	
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



//############################### #### ###############################
//############################### MAIN ###############################
//############################### #### ###############################

int main(){
	
	std::vector<size_t> sizes;
	int numsizes = 14;
	size_t tmp_size = 100;

	for(int i=0;i<numsizes; i+=2){
		sizes.push_back(tmp_size);
		sizes.push_back(3*tmp_size);
		tmp_size *= 10;
	}

	
	Timer mytimer, global_Timer;
	double elapsed_time;
	
	global_Timer.reset();
	

	int repetitions;

	bool do_task_a = true;
	bool do_task_b = true;
	bool do_task_c = true;



// -------------------------------- a ---------------------------------
// -------------------------------- a ---------------------------------
// -------------------------------- a ---------------------------------

	if (do_task_a) {
		repetitions = 1;
		printf("\n");
		printf("a) dot product stages 1 and 2 on GPU\n");
		printf("--------------------------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds; result\n");
		
		//int grid_size = GRID_SIZE;
		//int block_size = BLOCK_SIZE;
		
		for(int i=0; i<int(sizes.size()); i++){
			
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double));
			//double *result_stage1 = (double*)malloc(GRID_SIZE*sizeof(double));
			double result;
			
			double *gpu_x, *gpu_y, *gpu_result_stage1, *gpu_result_stage2;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_result_stage1, GRID_SIZE*sizeof(double));
			cudaMalloc(&gpu_result_stage2, sizeof(double));
			
			
			cpu_init(x,y,sizes[i]);
			
			cudaMemcpy(gpu_x,x,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y,y,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
									
			for(int j=0; j<repetitions; j++){
				gpu_dotproduct_stage1<<<GRID_SIZE,BLOCK_SIZE>>>(gpu_x,gpu_y,sizes[i], gpu_result_stage1);
				gpu_dotproduct_stage2<<<GRID_SIZE,BLOCK_SIZE>>>(gpu_result_stage1, gpu_result_stage2);		
				cudaMemcpy(&result,gpu_result_stage2,sizeof(double),cudaMemcpyDeviceToHost);
			}			

			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			
			printf("%1.0e;", double(sizes[i]));
			printf("%1.3e; ", elapsed_time);
			//printf("%1.8e\n", result);
			printf("%1.5e\n", result);

			cudaFree(gpu_x);
			cudaFree(gpu_y);
			cudaFree(gpu_result_stage1);
			free(x);
			free(y);
			//free(result_stage1);
		}
	}




// -------------------------------- b ---------------------------------
// -------------------------------- b ---------------------------------
// -------------------------------- b ---------------------------------

	if (do_task_b) {

		repetitions = 1;
		printf("\n");
		printf("b) dot product stage1 on GPU. sum on CPU\n");
		printf("--------------------------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds; result\n");
		
		//int grid_size = GRID_SIZE;
		//int block_size = BLOCK_SIZE;
		
		for(int i=0; i<int(sizes.size()); i++){
			
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double));
			double *result_stage1 = (double*)malloc(GRID_SIZE*sizeof(double));
			double result;
			
			double *gpu_x, *gpu_y, *gpu_result_stage1;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_result_stage1, GRID_SIZE*sizeof(double));
			
			cpu_init(x,y,sizes[i]);
			
			cudaMemcpy(gpu_x,x,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y,y,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
									
			for(int j=0; j<repetitions; j++){
				gpu_dotproduct_stage1<<<GRID_SIZE,BLOCK_SIZE>>>(gpu_x,gpu_y,sizes[i], gpu_result_stage1);
				cudaMemcpy(result_stage1,gpu_result_stage1,GRID_SIZE*sizeof(double),cudaMemcpyDeviceToHost);
				result = cpu_sum(result_stage1, GRID_SIZE);
			}

			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			
			printf("%1.0e;", double(sizes[i]));
			printf("%1.3e; ", elapsed_time);
			//printf("%1.8e\n", result);
			printf("%1.5e\n", result);

			cudaFree(gpu_x);
			cudaFree(gpu_y);
			cudaFree(gpu_result_stage1);
			free(x);
			free(y);
			free(result_stage1);
		}
	}


// -------------------------------- c ---------------------------------
// -------------------------------- c ---------------------------------
// -------------------------------- c ---------------------------------

	if (do_task_c) {
		repetitions = 1;
		printf("\n");
		printf("c) dot product on GPU with atomicAdd() \n");
		printf("--------------------------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds; result\n");
		
		//int grid_size = GRID_SIZE;
		//int block_size = BLOCK_SIZE;
		
		for(int i=0; i<int(sizes.size()); i++){
			
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double));
			//double *result_stage1 = (double*)malloc(GRID_SIZE*sizeof(double));
			double result;
			
			double *gpu_x, *gpu_y, *gpu_result;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			//cudaMalloc(&gpu_result_stage1, GRID_SIZE*sizeof(double));
			cudaMalloc(&gpu_result, sizeof(double));
						
			cpu_init(x,y,sizes[i]);
			
			cudaMemcpy(gpu_x,x,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y,y,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
									
			for(int j=0; j<repetitions; j++){
				gpu_dotproduct_atomicAdd<<<GRID_SIZE,BLOCK_SIZE>>>(gpu_x,gpu_y,sizes[i], gpu_result);
				cudaMemcpy(&result,gpu_result,sizeof(double),cudaMemcpyDeviceToHost);
			}			

			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			
			printf("%1.0e;", double(sizes[i]));
			printf("%1.3e; ", elapsed_time);
			//printf("%1.8e\n", result);
			printf("%1.5e\n", result);

			cudaFree(gpu_x);
			cudaFree(gpu_y);
			cudaFree(gpu_result);
			free(x);
			free(y);
			//free(result_stage1);
		}
	}


	printf("\n total elapsed time in seconds: %1.3e\n", global_Timer.get());		
	return 0;		
}
