#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_IMAGE

#include "sparse/SDRSPS.h"

#include "vis/PrettySDR.h"

#include <iostream>
#include <time.h>

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<sparse::SDRSPS> sdrsps(4);

	std::cout << "Initializing..." << std::endl;

	for (int i = 0; i < sdrsps.size(); i++)
		sdrsps[i].create(1000, 1000, 1000, 1000, 3, -0.01f, 0.01f, generator);
	
	std::cout << "Initialized." << std::endl;

	std::uniform_int_distribution<int> bitDist(0, 1000 * 1000 - 1);

	for (int iter = 0; iter < 10000; iter++) {
		std::vector<int> bits(1000);

		for (int i = 0; i < bits.size(); i++) {
			bits[i] = bitDist(generator);
		}

		sdrsps.front().activate(bits, 100, 100, 3, 0.001f);

		for (int i = 1; i < sdrsps.size(); i++) {
			sdrsps[i].activate(sdrsps[i - 1].getBitIndices(), 100, 100, 3, 0.001f);
		}
		std::cout << "Iteration " << iter << std::endl;
	}

	return 0;
}

#endif