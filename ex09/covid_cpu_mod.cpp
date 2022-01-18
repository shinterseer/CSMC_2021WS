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
  int    *contacts_per_day;           // number of other persons met each day to whom the disease may be passed on
  double *transmission_probability;   // how likely it is to pass on the infection to another person

} SimInput_t;

void init_input(SimInput_t *input)
{
  input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria

  input->mask_threshold      = 5000;
  input->lockdown_threshold  = 50000;
  input->infection_delay     = 5;     // 5 to 6 days incubation period (average) according to WHO
  input->infection_days      = 3;     // assume three days of passing on the disease
  input->starting_infections = 10;
  input->immunity_duration   = 180;   // half a year of immunity

  input->contacts_per_day = (int*)malloc(sizeof(int) * 365);
  input->transmission_probability = (double*)malloc(sizeof(double) * 365);
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


void run_simulation(const SimInput_t *input, SimOutput_t *output)
{
  //
  // Init data
  //
  for (int i=0; i<input->population_size; ++i) {
    output->is_infected[i] = (i < input->starting_infections) ? 1 : 0;
    output->infected_on[i] = (i < input->starting_infections) ? 0 : -1;
  }

  //
  // Run simulation
  //
	printf("day; lockdown; actively infected; recovered\n");	

  for (int day=0; day<365; ++day)  // loop over all days of the year
  {
    //
    // Step 1: determine number of infections and recoveries
    //
    int num_infected_current = 0;
    int num_recovered_current = 0;
    for (int i=0; i<input->population_size; ++i) {

      if (output->is_infected[i] > 0)
      {
        if (output->infected_on[i] > day - input->infection_delay - input->infection_days
           && output->infected_on[i] <= day - input->infection_delay)   // currently infected and incubation period over
          num_infected_current += 1;
        else if (output->infected_on[i] < day - input->infection_delay - input->infection_days)
          num_recovered_current += 1;
      }
    }

    output->active_infections[day] = num_infected_current;
    if (num_infected_current > input->lockdown_threshold) {
      output->lockdown[day] = 1;
    }
    if (day > 0 && output->lockdown[day-1] == 1) { // end lockdown if number of infections has reduced significantly
      output->lockdown[day] = (num_infected_current < input->lockdown_threshold / 3) ? 0 : 1;
    }
    char lockdown[] = " [LOCKDOWN]";
    char normal[] = "";
    // printf("Day %d%s: %d active, %d recovered\n", day, output->lockdown[day] ? lockdown : normal, num_infected_current, num_recovered_current);
    printf("%6i; %6i; %6i; %8i\n", day, output->lockdown[day], num_infected_current, num_recovered_current);


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
    for (int i=0; i<input->population_size; ++i) // loop over population
    {
      if (   output->is_infected[i] > 0
          && output->infected_on[i] > day - input->infection_delay - input->infection_days  // currently infected
          && output->infected_on[i] <= day - input->infection_delay)                         // already infectious
      {
        // pass on infection to other persons with transmission probability
        for (int j=0; j<contacts_today; ++j)
        {
          double r = ((double)rand()) / (double)RAND_MAX;  // random number between 0 and 1
          if (r < transmission_probability_today)
          {
            int other_person = r * input->population_size;
            if (output->is_infected[other_person] == 0
               || output->infected_on[other_person] < day - input->immunity_duration)
            {
              output->is_infected[other_person] = 1;
              output->infected_on[other_person] = day;
            }
          }

        } // for contacts_per_day
      } // if currently infected
    } // for i

  } // for day

}


int main(int argc, char **argv) {

  SimInput_t input;
  SimOutput_t output;

  init_input(&input);
  init_output(&output, input.population_size);

  Timer timer;
  srand(0); // initialize random seed for deterministic output
  timer.reset();
  run_simulation(&input, &output);
  printf("Simulation time: %g\n", timer.get());

  return EXIT_SUCCESS;
}