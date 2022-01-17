/**
 * 360.252 - Computational Science on Many-Core Architectures
 * WS 2021/22, TU Wien
 *
 * Simplistic COVID-19 simulator
 *
 * DISCLAIMER: This simulator is for educational purposes only.
 * It may be arbitrarily inaccurate and should not be used for drawing any conclusions about the actual COVID-19 virus.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.hpp"

#ifndef M_PI
  #define M_PI  3.14159265358979323846
#endif

#define GRID_SIZE 256
#define BLOCK_SIZE 256
// #define SIM_DAYS 365

#define POP_SIZE 8916845
// #define POP_SIZE 10000
#define RP_SIZE 10000

#define R_MAX 4294967296
#define R_A 1664525
#define R_C 1013904223


//
// Data container for simulation input
//
typedef struct
{

  size_t population_size;    // Number of people to simulate

  //// Configuration
  int mask_threshold;        // Number of cases required for masks
  int lockdown_threshold;    // Number of cases required for lockdown
  int infection_delay;       // Number of days before an infected person can pass on the disease
  int infection_days;        // Number of days an infected person can pass on the disease
  int starting_infections;   // Number of infected people at the start of the year
  int immunity_duration;     // Number of days a recovered person is immune

  // for each day:
  int    contacts_per_day[365];           // number of other persons met each day to whom the disease may be passed on
  double transmission_probability[365];   // how likely it is to pass on the infection to another person

} SimInput_t;

void init_input(SimInput_t *input)
{
  input->population_size = POP_SIZE;  // Austria's population in 2020 according to Statistik Austria

  input->mask_threshold      = 5000;
  input->lockdown_threshold  = 50000;
  input->infection_delay     = 5;     // 5 to 6 days incubation period (average) according to WHO
  input->infection_days      = 3;     // assume three days of passing on the disease
  input->starting_infections = 10;
  input->immunity_duration   = 180;   // half a year of immunity

  for (int day = 0; day < 365; ++day) {
    input->contacts_per_day[day] = 6;             // arbitrary assumption of six possible transmission contacts per person per day, all year
    input->transmission_probability[day] = 0.2
                                           + 0.1 * cos((day / 365.0) * 2 * M_PI);   // higher transmission in winter, lower transmission during summer
  }
}


typedef struct
{
  // for each day:
  int *active_infections;     // number of active infected on that day (including incubation period)
  int *lockdown;              // 0 if no lockdown on that day, 1 if lockdown

  // for each person:
  int *is_infected;      // 0 if healty, 1 if currently infected
  int *infected_on;      // day of infection. negative if not yet infected. January 1 is Day 0.

} SimOutput_t;

void init_output(SimOutput_t *output, int population_size)
{
  output->active_infections = (int*)malloc(sizeof(int) * 365);
  output->lockdown          = (int*)malloc(sizeof(int) * 365);
  for (int day = 0; day < 365; ++day) {
    output->active_infections[day] = 0;
    output->lockdown[day] = 0;
  }

  output->is_infected       = (int*)malloc(sizeof(int) * population_size);
  output->infected_on       = (int*)malloc(sizeof(int) * population_size);

  for (int i=0; i<population_size; ++i) {
    output->is_infected[i] = 0;
    output->infected_on[i] = 0;
  }
}



__global__ 
void step1_gpu(int day, SimInput_t *device_input, int* device_is_infected, int* device_infected_on,
							 int* device_num_infected_current, int* device_num_recovered_current){
	
	int thread_idx_global = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;
	
	__shared__ int local_infected[BLOCK_SIZE];
	__shared__ int local_recovered[BLOCK_SIZE];
	
	local_infected[threadIdx.x] = 0;
	local_recovered[threadIdx.x] = 0;
		
	for (int i = thread_idx_global; i < device_input->population_size; i += num_threads) {	
		if (device_is_infected[i] > 0)
		{
			if (device_infected_on[i] > day - device_input->infection_delay - device_input->infection_days
				 && device_infected_on[i] <= day - device_input->infection_delay)   // currently infected and incubation period over
				local_infected[threadIdx.x] += 1;
			else if (device_infected_on[i] < day - device_input->infection_delay - device_input->infection_days)
				local_recovered[threadIdx.x] += 1;
		}
	}
	
	// now the reduction
	for(int stride = blockDim.x/2; stride>0; stride/=2){
		__syncthreads();
		if (threadIdx.x < stride){
			local_infected[threadIdx.x] += local_infected[threadIdx.x + stride];
			local_recovered[threadIdx.x] += local_recovered[threadIdx.x + stride];
		}
	}
		
	if (threadIdx.x == 0){
		atomicAdd(device_num_infected_current, local_infected[0]);
		atomicAdd(device_num_recovered_current, local_recovered[0]);		
	}
}


__global__
void step3_gpu(int day, SimInput_t *device_input, int* device_is_infected, int* device_infected_on,
							 int contacts_today, double transmission_probability_today, int* device_random_number,
							 int *device_num_infected_current, int *device_num_recovered_current){

	int thread_idx_global = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;

	unsigned int random_int;	
	double r;
	
	for (int i = thread_idx_global; i < device_input->population_size; i += num_threads) // loop over population
	{
		if (   device_is_infected[i] > 0
				&& device_infected_on[i] > day - device_input->infection_delay - device_input->infection_days  // currently infected
				&& device_infected_on[i] <= day - device_input->infection_delay)                         // already infectious
		{
			// pass on infection to other persons with transmission probability
			for (int j = 0; j < contacts_today; ++j)
			{
				random_int = (R_A*device_random_number[thread_idx_global] + R_C) % R_MAX;
				device_random_number[thread_idx_global] = random_int;
				r = double(random_int) / R_MAX;
				
				if (r < transmission_probability_today)
				{
					int other_person = r * device_input->population_size;
					if (device_is_infected[other_person] == 0
						 || device_infected_on[other_person] < day - device_input->immunity_duration)
					{
						device_is_infected[other_person] = 1;
						device_infected_on[other_person] = day;
					}
				}
			} // for contacts_per_day
		} // if currently infected
	} // for i
	
	// initialize for other kernel
	if (thread_idx_global == 0){
		*device_num_infected_current = 0;
		*device_num_recovered_current = 0;
	}

}


void run_simulation(const SimInput_t *input, SimOutput_t *output, int sim_days = 365)
{
  //
  // Init data
  //
  for (int i=0; i<input->population_size; ++i) {
    output->is_infected[i] = (i < input->starting_infections) ? 1 : 0;
    output->infected_on[i] = (i < input->starting_infections) ? 0 : -1;
  }
	int num_infected_current = 0;
	int num_recovered_current = 0;
	double random_pool[RP_SIZE];
	for(int i = 0; i < RP_SIZE; ++i)
		random_pool[i] = ((double)rand()) / (double)RAND_MAX;  // random number between 0 and 1

  //
  // allocate on gpu
  //
	int *device_is_infected, *device_infected_on;
	cudaMalloc(&device_is_infected, input->population_size*sizeof(int));
	cudaMalloc(&device_infected_on, input->population_size*sizeof(int));
	int *device_num_infected_current, *device_num_recovered_current;
	cudaMalloc(&device_num_infected_current, sizeof(int));
	cudaMalloc(&device_num_recovered_current, sizeof(int));

	int *device_random_number;
	cudaMalloc(&device_random_number, GRID_SIZE*BLOCK_SIZE*sizeof(int));

	// initialize random numbers on cpu
	int random_number[GRID_SIZE*BLOCK_SIZE];
	for(int i = 0; i < GRID_SIZE*BLOCK_SIZE; ++i)
		random_number[i] = i;

	
	SimInput_t *device_input;	
	cudaMalloc(&device_input, sizeof(SimInput_t));
	
	//
  // move data to gpu
  //	
	cudaMemcpy(device_input, input, sizeof(SimInput_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_random_number, random_number, GRID_SIZE*BLOCK_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_is_infected, output->is_infected, input->population_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_infected_on, output->infected_on, input->population_size*sizeof(int), cudaMemcpyHostToDevice);

	
  //
  // Run simulation
  //
  for (int day=0; day<sim_days; ++day)  // loop over all days of the year
  {
    //
    // Step 1: determine number of infections and recoveries
    //
		step1_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(day, device_input, device_is_infected, device_infected_on,
												device_num_infected_current, device_num_recovered_current);
		// copy back num_infected_current and num_recovered_current
		cudaMemcpy(&num_infected_current, device_num_infected_current, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&num_recovered_current, device_num_recovered_current, sizeof(int), cudaMemcpyDeviceToHost);
	

    output->active_infections[day] = num_infected_current;
    if (num_infected_current > input->lockdown_threshold) {
      output->lockdown[day] = 1;
    }
    if (day > 0 && output->lockdown[day-1] == 1) { // end lockdown if number of infections has reduced significantly
      output->lockdown[day] = (num_infected_current < input->lockdown_threshold / 3) ? 0 : 1;
    }
    char lockdown[] = " [LOCKDOWN]";
    char normal[] = "";
    printf("Day %d%s: %d active, %d recovered\n", day, output->lockdown[day] ? lockdown : normal, num_infected_current, num_recovered_current);


    //
    // Step 2: determine today's transmission probability and contacts based on pandemic situation
    //
    double contacts_today = input->contacts_per_day[day];
    double transmission_probability_today = input->transmission_probability[day];
    if (num_infected_current > input->mask_threshold) { // transmission is reduced with masks. Arbitrary factor: 2
      transmission_probability_today /= 2.0;
    }
    if (output->lockdown[day]) { // contacts are significantly reduced in lockdown. Arbitrary factor: 4
      contacts_today /= 4;
    }

    //
    // Step 3: pass on infections within population
    //
		step3_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(day, device_input, device_is_infected, device_infected_on,
								 contacts_today, transmission_probability_today, device_random_number,
								 device_num_recovered_current, device_num_infected_current);

  } // for day

	cudaFree(device_input);
	cudaFree(device_is_infected);
	cudaFree(device_infected_on);
	cudaFree(device_num_infected_current);
	cudaFree(device_num_recovered_current);
	cudaFree(device_random_number);
	
}



int main(int argc, char **argv) {

  SimInput_t input;
  SimOutput_t output;

  init_input(&input);
  init_output(&output, input.population_size);

  Timer timer;
  srand(0); // initialize random seed for deterministic output
  timer.reset();
  run_simulation(&input, &output, 365);
  printf("Simulation time: %g\n", timer.get());

	cudaDeviceReset();  // for CUDA leak checker to work

  return EXIT_SUCCESS;
}