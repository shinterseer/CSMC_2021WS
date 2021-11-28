#include <stdio.h>
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


int main(){
	
	int sizes[7] = {1000, int(1e4), int(1e5), int(1e6), int(1e7), int(1e8), int(1e9)};
	double elapsed_time;

	/*
	// dummy allocations - the gpu seems to take about 1.5s for the first allocation
	double *dummy, *dommy, *timmy;
	cudaMalloc(&dummy, 100*sizeof(double));
	cudaMalloc(&dommy, 100*sizeof(double));
	cudaMalloc(&timmy, 100*sizeof(double));
	*/


	printf("size; elapsed time in seconds (allocation); elapsed time in seconds (free); \n");
	Timer mytimer;
	//Timer mytimer_free;
	for(int i=0;i<7;i++){
		double *gpu_x, *gpu_y;
		
		cudaDeviceSynchronize(); //make sure gpu is ready
		mytimer.reset();
		
		cudaMalloc(&gpu_x, sizes[i]*sizeof(double));
		cudaMalloc(&gpu_y, sizes[i]*sizeof(double));
		
		cudaDeviceSynchronize(); //make sure gpu has finished
		
		elapsed_time = mytimer.get();
		printf("%i; %1.3e; ",sizes[i], elapsed_time);		
		
		cudaDeviceSynchronize(); //make sure gpu is ready
		mytimer.reset();
		
		cudaFree(gpu_x);
		cudaFree(gpu_y);

		elapsed_time = mytimer.get();
		printf("%1.3e\n",elapsed_time);		

	}
	
	return 0;	
}
