#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_IMAGE

#include "sparse/PredictiveHierarchy.h"

#include "vis/PrettySDR.h"

#include <iostream>
#include <time.h>

int main() {
	std::mt19937 generator(time(nullptr));

	/*std::vector<sparse::SDRSPS> sdrsps(4);

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
	}*/

	/*sf::RenderWindow rw;

	rw.create(sf::VideoMode(800, 600), "Test");

	sf::Clock clock;

	const float frameTime = 1.0f / 60.0f;
	const float renderMouseTime = 0.001f;

	bool quit = false;

	while (!quit) {
		clock.restart();

		sf::Event e;

		while (rw.pollEvent(e)) {
			switch (e.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		rw.clear();

		float ts = clock.getElapsedTime().asSeconds();

		while (ts < frameTime - renderMouseTime) {
			ts = clock.getElapsedTime().asSeconds();
		}

		sf::CircleShape cs;

		sf::Vector2i mousePos = sf::Mouse::getPosition(rw);

		cs.setFillColor(sf::Color::Blue);
		cs.setRadius(3.0f);

		cs.setPosition(sf::Vector2f(mousePos.x, mousePos.y));

		rw.draw(cs);

		rw.display();
	}*/

	std::vector<sparse::PredictiveHierarchy::LayerDesc> layerDescs(1);

	sparse::PredictiveHierarchy ph;

	ph.create(4, 4, layerDescs, -0.01f, 0.01f, generator);

	std::vector<std::vector<sparse::BitIndexType>> bits = {
		{ 0, 4, 9 },
		{ 1, 4, 5 },
		{ 4, 7, 9 },
		{ 3, 6, 9 }
	};

	for (int iter = 0; iter < 1000; iter++) {
		ph.simStep(bits[iter % bits.size()]);

		for (int i = 0; i < ph.getPredBitIndices().size(); i++)
			std::cout << ph.getPredBitIndices()[i] << " ";

		//for (int i = 0; i < ph.getLayers()[2]._pred.getBitIndices().size(); i++)
		//	std::cout << ph.getLayers()[2]._pred.getBitIndices()[i] << " ";

		std::cout << std::endl;
	}

	return 0;
}

#endif