// _______________________________ Ex4 Kernels ____________________________
// ________________________________________________________________________



__global__
void gpu_dotproduct_atomicAdd(const double *gpu_x, const double *gpu_y, const size_t size, double *gpu_result){
	
	size_t thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread_id_global == 0)
		*gpu_result = 0;
	
	__shared__ double shared_m[BLOCK_SIZE];
		
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
	// thread 0 of the workgroup writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_result, shared_m[0]);		
	}	
}


__global__
void gpu_stuff_sharedm(const double *gpu_x, const size_t size, 
										  double *gpu_sum, double *gpu_abssum, double *gpu_squaresum, double *gpu_numzeros){
	
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;

	__shared__ double shared_sum[BLOCK_SIZE];
	__shared__ double shared_abssum[BLOCK_SIZE];
	__shared__ double shared_squaresum[BLOCK_SIZE];
	__shared__ double shared_numzeros[BLOCK_SIZE];
		
	// sum of entries
	double thread_sum = 0;
	double thread_abssum = 0;
	double thread_squaresum = 0;
	double thread_numzeros = 0;
	
	if (thread_id_global == 0){
		*gpu_sum = 0;
		*gpu_abssum = 0;
		*gpu_squaresum = 0;
		*gpu_numzeros = 0;
	}
		
	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_sum += gpu_x[i];
		thread_abssum += std::abs(gpu_x[i]);
		thread_squaresum += gpu_x[i] * gpu_x[i];
		if (gpu_x[i] == 0)
			thread_numzeros += 1.;	
	}

	shared_sum[threadIdx.x] = thread_sum;
	shared_abssum[threadIdx.x] = thread_abssum;
	shared_squaresum[threadIdx.x] = thread_squaresum;
	shared_numzeros[threadIdx.x] = thread_numzeros;
		
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
			shared_abssum[threadIdx.x] += shared_abssum[threadIdx.x + stride];
			shared_squaresum[threadIdx.x] += shared_squaresum[threadIdx.x + stride];
			shared_numzeros[threadIdx.x] += shared_numzeros[threadIdx.x + stride];			
		}
	}
	
	__syncthreads();
	// thread 0 writes result
	if (threadIdx.x == 0){
		atomicAdd(gpu_sum, shared_sum[0]);
		atomicAdd(gpu_abssum, shared_abssum[0]);
		atomicAdd(gpu_squaresum, shared_squaresum[0]);
		atomicAdd(gpu_numzeros, shared_numzeros[0]);
	}	
}


__global__
void gpu_stuff_wshuffle(const double *gpu_x, const size_t size, 
										  double *gpu_sum, double *gpu_abssum, double *gpu_squaresum, double *gpu_numzeros){
		
	int thread_id_global = blockIdx.x*blockDim.x + threadIdx.x;
	int thread_num = gridDim.x * blockDim.x;
	
	// sum of entries
	double thread_sum = 0;
	double thread_abssum = 0;
	double thread_squaresum = 0;
	double thread_numzeros = 0;
	
	if (thread_id_global == 0){
		*gpu_sum = 0;
		*gpu_abssum = 0;
		*gpu_squaresum = 0;
		*gpu_numzeros = 0;
	}

	for (unsigned int i = thread_id_global; i < size; i += thread_num){
		thread_sum += gpu_x[i];
		thread_abssum += std::abs(gpu_x[i]);
		thread_squaresum += gpu_x[i] * gpu_x[i];
		if (gpu_x[i] == 0)
			thread_numzeros += 1.;	
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
		thread_sum += __shfl_down_sync(ALL_MASK, thread_sum, shuffle_delta);
		thread_abssum += __shfl_down_sync(ALL_MASK, thread_abssum, shuffle_delta);
		thread_squaresum += __shfl_down_sync(ALL_MASK, thread_squaresum, shuffle_delta);
		thread_numzeros += __shfl_down_sync(ALL_MASK, thread_numzeros, shuffle_delta);
	}
	
	__syncwarp();
	// thread 0 (of each warp) writes result
	if ((threadIdx.x % warpSize) == 0){
		atomicAdd(gpu_sum, thread_sum);
		atomicAdd(gpu_abssum, thread_abssum);
		atomicAdd(gpu_squaresum, thread_squaresum);
		atomicAdd(gpu_numzeros, thread_numzeros);
	}	
}


// _______________________________ Ex2 Kernels ____________________________
// ________________________________________________________________________


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
void gpu_dotproduct_atomicAdd2(const double *gpu_x, const double *gpu_y, size_t size, double *gpu_result){
	
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
