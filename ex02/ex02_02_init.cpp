	#include <stdio.h>
	#include <vector>
	#include "timer.hpp"


	void init_arrays(double *x, double *y, int size){
		for(int i=0; i<size; i++){
			x[i] = i;
			y[i] = size - i - 1;		
		}
	}


	void cpuwork(int amount){
		volatile int blub = 0;
		for(int i=0; i<amount;i++){
			blub++;
			blub--;
		}
	}

	__global__ 
	void gpu_init(double *gpu_x, double *gpu_y, int size){

		int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		for(size_t i=thread_id; i<size; i += blockDim.x*gridDim.x){
		//for(int i=0; i<size; i++){
			gpu_x[i] = i;
			gpu_y[i] = size - i - 1;
		}
	}

	void cpu_init(double *x, double *y, int size){
		for(int i=0; i<size; i++){
			x[i] = i;
			y[i] = size - i - 1;
		}
	}



	int main(){
		
		//int sizes[7] = {1000, int(1e4), int(1e5), int(1e6), int(1e7), int(1e8), int(1e9)};
		//int sizes[11] = {100, 300,  1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000};
		std::vector<int> sizes;		
		Timer mytimer, global_Timer;
		double elapsed_time;
		
		global_Timer.reset();
		
		int numsizes = 12;
		int tmp_size = 100;

		for(int i=0;i<numsizes; i+=2){
			sizes.push_back(tmp_size);
			sizes.push_back(3*tmp_size);
			tmp_size *= 10;
		}
		
		printf("sizes.size = %i\n", sizes.size());
		

		/*
		// dummy allocations - the gpu seems to take about 1.5s for the first allocation
		double *dummy, *dommy, *timmy;
		cudaMalloc(&dummy, 100*sizeof(double));
		cudaMalloc(&dommy, 100*sizeof(double));
		cudaMalloc(&timmy, 100*sizeof(double));
		*/
		
		int repetitions;

		repetitions = 1;
		printf("\n");
		printf("initializing on GPU \n");
		printf("------------------- \n");	
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds;\n");
		for(int i=0; i<int(sizes.size()); i++){
			double *gpu_x, *gpu_y;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cudaDeviceSynchronize(); //make sure gpu is ready

			mytimer.reset();
			for(int j=0; j<repetitions; j++){
				gpu_init<<<256,256>>>(gpu_x,gpu_y,sizes[i]);
			}
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();

			printf("%i;", sizes[i]);
			printf("%1.3e\n", elapsed_time);

			cudaFree(gpu_x);
			cudaFree(gpu_y);		
		}

		repetitions = 1;
		printf("\n");
		printf("initializing on CPU \n");
		printf("------------------- \n");
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds;\n");
		
		for(int i=0; i<int(sizes.size())-2; i++){
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double)); 
			
			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
			for(int j=0; j<repetitions; j++){
				cpu_init(x,y,sizes[i]);
			}		
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			printf("%i;", sizes[i]);
			printf("%1.3e\n", elapsed_time);

			free(x);
			free(y);
			
		}

		repetitions = 1;
		printf("\n");
		printf("copy it over (assume already initialized) \n");
		printf("------------------- \n");
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds;\n");
		
		for(int i=0; i<int(sizes.size()); i++){
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double)); 
			double *gpu_x, *gpu_y;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cpu_init(x,y,sizes[i]);

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
			for(int j=0; j<repetitions; j++){
				cudaMemcpy(gpu_x,x,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_y,y,sizes[i]*sizeof(double),cudaMemcpyHostToDevice);
			}
			
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			printf("%i;", sizes[i]);
			printf("%1.3e\n", elapsed_time);		

			free(x);
			free(y);
			cudaFree(gpu_x);
			cudaFree(gpu_y);
		}

		
		
		repetitions = 1;
		printf("\n");
		printf("individual transfer (assume already initialized) \n");
		printf("------------------- \n");
		printf("------------------- \n");
		printf("repetitions = %i \n",repetitions);
		printf("size; elapsed time in seconds;\n");
		
		for(int i=0; i<int(sizes.size())-4; i++){
			double *x = (double*)malloc(sizes[i]*sizeof(double)); 
			double *y = (double*)malloc(sizes[i]*sizeof(double)); 
			double *gpu_x, *gpu_y;
			cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
			cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
			cpu_init(x,y,sizes[i]);

			cudaDeviceSynchronize(); //make sure gpu is ready
			mytimer.reset();
			for(int j=0; j<repetitions; j++){
				for(int k=0; k<sizes[i]; k++){
					//cudaMemcpy(&x[k],&gpu_x[k],sizeof(double),cudaMemcpyHostToDevice);
					//cudaMemcpy(&y[k],&gpu_y[k],sizeof(double),cudaMemcpyHostToDevice);				
					cudaMemcpy(&gpu_x[k],&x[k],sizeof(double),cudaMemcpyHostToDevice);
					cudaMemcpy(&gpu_y[k],&y[k],sizeof(double),cudaMemcpyHostToDevice);				
				}
			}		
			cudaDeviceSynchronize(); //make sure gpu has finished
			elapsed_time = mytimer.get();
			printf("%i;", sizes[i]);
			printf("%1.3e\n", elapsed_time);		

			free(x);
			free(y);		
			cudaFree(gpu_x);
			cudaFree(gpu_y);
		}		

		printf("\n total elapsed time in seconds: %1.3e\n", global_Timer.get());		
		return 0;
		
	}
