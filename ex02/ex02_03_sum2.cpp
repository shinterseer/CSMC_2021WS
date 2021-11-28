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
	
	std::vector<size_t> sizes = {size_t(1e7)};
	Timer mytimer, global_Timer;
	double elapsed_time;
	
	global_Timer.reset();
	
	/*int numsizes = 14;
	size_t tmp_size = 100;
	

	for(int i=0;i<numsizes; i+=2){
		sizes.push_back(tmp_size);
		sizes.push_back(3*tmp_size);
		tmp_size *= 10;
	}
	*/
	
/*		int grids[5] = {16, 32, 64, 128, 256};
	int numgrids = 5;
	int blocks[5] = {16, 32, 64, 128, 256};
	int numblocks = 5;
	
	int grids[4] = {32, 64, 128, 256};
	int numgrids = 4;
	int blocks[4] = {32, 64, 128, 256};
	int numblocks = 4;
*/

/*		int grids[5] = {64, 128, 256, 512, 1024};
	int numgrids = 5;
	int blocks[5] = {64, 128, 256, 512, 1024};
	int numblocks = 5;
*/
	
	int grids[7] = {16, 32, 64, 128, 256, 512, 1024};
	int numgrids = 7;
	int blocks[7] = {16, 32, 64, 128, 256, 512, 1024};
	int numblocks = 7;


	int repetitions;

	repetitions = 1;
	printf("\n");
	printf("summing vectors on GPU \n");
	printf("------------------- \n");	
	printf("repetitions = %i \n",repetitions);
	printf("gridsize; blocksize; elapsed time in seconds;\n");
	for(int i=0; i<numgrids; i++){
		for(int j=0; j<numblocks; j++){
			
			double *x = (double*)malloc(sizes[0]*sizeof(double)); 
			double *y = (double*)malloc(sizes[0]*sizeof(double)); 
			double *result = (double*)malloc(sizes[0]*sizeof(double)); 
			
			double *gpu_x, *gpu_y, *gpu_result;
			cudaMalloc(&gpu_x, sizes[0]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[0]*sizeof(double));
			cudaMalloc(&gpu_result, sizes[0]*sizeof(double));
			
			cpu_init(x,y,sizes[0]);
			
			cudaMemcpy(gpu_x,x,sizes[0]*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y,y,sizes[0]*sizeof(double),cudaMemcpyHostToDevice);	

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
									
			for(int ii=0; ii<repetitions; ii++){
				gpu_sum<<<grids[i],blocks[j]>>>(gpu_x,gpu_y,gpu_result,sizes[0]);
			}
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();

			cudaMemcpy(result,gpu_result,sizes[0]*sizeof(double),cudaMemcpyDeviceToHost);

			//show_result(result);
			
			printf("%i;", grids[i]);
			printf("%i;", blocks[j]);
			printf("%1.3e\n", elapsed_time);

			cudaFree(gpu_x);
			cudaFree(gpu_y);
			cudaFree(gpu_result);
			free(x);
			free(y);
			free(result);
		}
	}

	printf("\n total elapsed time in seconds: %1.3e\n", global_Timer.get());		
	return 0;		
}
