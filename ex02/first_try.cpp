#include <stdio.h>
#include "timer.hpp"

void cpuwork(int amount){
	volatile int blub = 0;
	for(int i=0; i<amount;i++){
		blub++;
		blub--;
	}
}




int main(){
	//printf("Hello World!");
	int amount = 1e9;
	Timer mytimer;
	mytimer.reset();
	cpuwork(amount);
	printf("finished cpu work (amount = %1.2e)\n", double(amount));
	printf("elapsed time = %4.3f seconds", mytimer.get());
	return 0;
	
}
