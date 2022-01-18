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
// #define POP_SIZE 100

// random number generator constants



// from wikipedia (numerical recipes)
#define R_MAX1 4294967296
#define R_A1 1664525
#define R_C1 1013904223

// from wikipedia (borland C/C++) mod
#define R_MAX2 4294967290
#define R_A2 22695477
#define R_C2 1

// wikipedia (ZX81)
#define R_MAX3 65537
#define R_A3 75
#define R_C3 74

// least common multiple seeems to be
// according to https://www.calculatorsoup.com/calculators/math/lcd.php
// 9222527599039741952 ~= 2^63

// from wikipedia (borland Delphi, Virtual Pascal) mod
// #define R_MAX3 4294967293
// #define R_A3 134775813
// #define R_C3 1

// from wikipedia (borland C/C++)
// #define R_MAX2 4294967296
// #define R_A2 22695477
// #define R_C2 1

// from wikipedia (borland Delphi, Virtual Pascal)
// #define R_MAX3 4294967296
// #define R_A3 134775813
// #define R_C3 1

// awful constants from wu.ac.at
// #define R_MAX 2147483647
// #define R_A 950706376  
// #define R_C 0


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
	double lockdown_discipline; // chance that people will respect a lockdown

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
	input->lockdown_discipline = 1; // 1: lockdowns are always respected, 0: ld always ignored

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
  // int *is_infected;      // 0 if healty, 1 if currently infected
  // int *infected_on;      // day of infection. negative if not yet infected. January 1 is Day 0.

} SimOutput_t;

// void init_output(SimOutput_t *output, int population_size)
void init_output(SimOutput_t *output)
{
  output->active_infections = (int*)malloc(sizeof(int) * 365);
  output->lockdown          = (int*)malloc(sizeof(int) * 365);
  for (int day = 0; day < 365; ++day) {
    output->active_infections[day] = 0;
    output->lockdown[day] = 0;
  }

  // output->is_infected       = (int*)malloc(sizeof(int) * population_size);
  // output->infected_on       = (int*)malloc(sizeof(int) * population_size);

  // for (int i=0; i<population_size; ++i) {
    // output->is_infected[i] = 0;
    // output->infected_on[i] = 0;
  // }
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

__device__
inline void myLCG1(unsigned int *address){
	*address = (R_A1*(*address) + R_C1) % R_MAX1;
}

__device__
inline void myLCG2(unsigned int *address){
	*address = (R_A2*(*address) + R_C2) % R_MAX2;
}

__device__
inline void myLCG3(unsigned int *address){
	*address = (R_A3*(*address) + R_C3) % R_MAX3;
}

__device__
inline double myRNG(unsigned int *first_address){	
  // This combines three LCG
	// update random integers
	myLCG1(first_address+0);
	myLCG2(first_address+1);
	myLCG3(first_address+2);
	// get sum of random numbers (each between 0 and 1)
  double r = double(*(first_address+0))/R_MAX1 + 
						 double(*(first_address+1))/R_MAX2 + 
						 double(*(first_address+2))/R_MAX3;
	
	// return fmod(r,1);
	return r - int(r);
}

__global__
void step3_gpu(int day, SimInput_t *device_input, int* device_is_infected, int* device_infected_on,
							 double contacts_today, double transmission_probability_today, unsigned int *device_random_number,
							 int *device_num_infected_current, int *device_num_recovered_current){

	int thread_idx_global = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;

	// unsigned int random_int;	
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
				r = myRNG(&device_random_number[3*thread_idx_global]);
				
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

__global__
void step3_gpu_mod(int day, SimInput_t *device_input, int* device_is_infected, int* device_infected_on,
							 int lockdown, double transmission_probability_today, unsigned int *device_random_number,
							 int *device_num_infected_current, int *device_num_recovered_current){

	int thread_idx_global = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;

	unsigned int random_int;	
	double r;
	
	double contact_ratio = 1.0;
	if (lockdown)
		contact_ratio = 0.25;	
	
	for (int i = thread_idx_global; i < device_input->population_size; i += num_threads) // loop over population
	{
		if (   device_is_infected[i] > 0
				&& device_infected_on[i] > day - device_input->infection_delay - device_input->infection_days  // currently infected
				&& device_infected_on[i] <= day - device_input->infection_delay)                         // already infectious
		{
			// roll the dice, whether the person will respect a lockdown
			r = myRNG(&device_random_number[3*thread_idx_global]);
			if (r > device_input->lockdown_discipline)
				contact_ratio = 1.0;


			// pass on infection to other persons with transmission probability
			for (int j = 0; j < (device_input->contacts_per_day[day])*contact_ratio; ++j)
			{
				r = myRNG(&device_random_number[3*thread_idx_global]);

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

__global__
void init_gpu(int *device_is_infected, int *device_infected_on, 
							int *device_num_infected_current, int *device_num_recovered_current, unsigned int *device_random_number,
							int population_size, int starting_infections)
{
	int thread_idx_global = blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;
		
	for (int i = thread_idx_global; i < population_size; i += num_threads){
		// init large arrays
    device_is_infected[i] = (i < starting_infections) ? 1 : 0;
    device_infected_on[i] = (i < starting_infections) ? 0 : -1;
  }

	// init random number seeds
  device_random_number[3*thread_idx_global] = thread_idx_global;
  device_random_number[3*thread_idx_global+1] = thread_idx_global;
  device_random_number[3*thread_idx_global+2] = thread_idx_global;

	// init for first iteration
	if (thread_idx_global == 0){
		*device_num_infected_current = 0;
		*device_num_recovered_current = 0;
	}	
}


// void run_simulation(const SimInput_t *input, SimOutput_t *output, int sim_days = 365)
void run_simulation(int sim_days = 365)
{
  SimInput_t input;
  SimOutput_t output;

  init_input(&input);
  init_output(&output);
	
	int num_infected_current;
	int num_recovered_current;

  //
  // allocate on gpu
  //
	int *device_is_infected, *device_infected_on;
	cudaMalloc(&device_is_infected, input.population_size*sizeof(int));
	cudaMalloc(&device_infected_on, input.population_size*sizeof(int));
	
	int *device_num_infected_current, *device_num_recovered_current;
	cudaMalloc(&device_num_infected_current, sizeof(int));
	cudaMalloc(&device_num_recovered_current, sizeof(int));

	unsigned int *device_random_number;
	cudaMalloc(&device_random_number, GRID_SIZE*BLOCK_SIZE*3*sizeof(unsigned int));

	cudaDeviceSynchronize();
	init_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(device_is_infected,device_infected_on,
																		 device_num_infected_current,device_num_recovered_current, device_random_number,
																		 input.population_size,input.starting_infections);
	cudaDeviceSynchronize();

	
	SimInput_t *device_input;	
	cudaMalloc(&device_input, sizeof(SimInput_t));
	
	//
  // move data to gpu
  //	
	cudaMemcpy(device_input, &input, sizeof(SimInput_t), cudaMemcpyHostToDevice);
	
  //
  // Run simulation
  //
	printf("day; lockdown; actively infected; recovered\n");	
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
	

    output.active_infections[day] = num_infected_current;
    if (num_infected_current > input.lockdown_threshold) {
      output.lockdown[day] = 1;
    }
    if (day > 0 && output.lockdown[day-1] == 1) { // end lockdown if number of infections has reduced significantly
      output.lockdown[day] = (num_infected_current < input.lockdown_threshold / 3) ? 0 : 1;
    }
    printf("%6i; %6i; %6i; %8i\n", day, output.lockdown[day], num_infected_current, num_recovered_current);


    //
    // Step 2: determine today's transmission probability and contacts based on pandemic situation
    //
    double contacts_today = input.contacts_per_day[day];
    double transmission_probability_today = input.transmission_probability[day];
    if (num_infected_current > input.mask_threshold) { // transmission is reduced with masks. Arbitrary factor: 2
      transmission_probability_today /= 2.0;
    }
    if (output.lockdown[day]) { // contacts are significantly reduced in lockdown. Arbitrary factor: 4
      contacts_today /= 4;
    }

    //
    // Step 3: pass on infections within population
    //
		// step3_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(day, device_input, device_is_infected, device_infected_on,
								 // contacts_today, transmission_probability_today, device_random_number,
								 // device_num_recovered_current, device_num_infected_current);
		step3_gpu_mod<<<GRID_SIZE,BLOCK_SIZE>>>(day, device_input, device_is_infected, device_infected_on,
								 output.lockdown[day], transmission_probability_today, device_random_number,
								 device_num_recovered_current, device_num_infected_current);


  } // for day

	cudaFree(device_input);
	cudaFree(device_is_infected);
	cudaFree(device_infected_on);
	cudaFree(device_num_infected_current);
	cudaFree(device_num_recovered_current);
	cudaFree(device_random_number);
}


void rng_test(int reps = 100000){
	unsigned int seed = 0;
	unsigned int r_int = (R_A1*seed + R_C1) % R_MAX1;

	constexpr int num_bins = 100;
	int bins[num_bins] = {0};

	double r;
	for(int i = 0; i < reps; ++i){
		r_int = (R_A1*r_int + R_C1) % R_MAX1;
		r = double(r_int) / R_MAX1;
		// printf("r: %2.3f\n",r);
		
		for(int j = 0; j < num_bins; ++j){
			if ((r >= j*1.0/num_bins) && (r < (j+1)*1.0/num_bins))
				bins[j]++;
		}		
	}
	
	for(int i = 0; i < num_bins; ++i)
		printf("bin index: %3i; number of random numbers: %5i\n",i,bins[i]);	
}


/*
 unsigned z1, z2, z3, z4; 
 float HybridTaus() {   
 // Combined period is lcm(p1,p2,p3,p4)~ 2^121    
 return 2.3283064365387e-10 * (              
 // Periods      TausStep(z1, 13, 19, 12, 4294967294UL) ^   // p1=2^31-1     TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1     TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1     LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32    ); }

*/

int main(int argc, char **argv) {

  // SimInput_t input;
  // SimOutput_t output;

  // init_input(&input);
  // init_output(&output, input.population_size);

  Timer timer;
  timer.reset();
	// rng_test();
  run_simulation();
  printf("Simulation time: %g\n", timer.get());

	cudaDeviceReset();  // for CUDA leak checker to work

  return EXIT_SUCCESS;
}