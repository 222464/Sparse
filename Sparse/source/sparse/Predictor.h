#pragma once

#include "SDRSPS.h"

namespace sparse {
	class Predictor {
	public:
		struct Connection {
			WeightType _weight;
		};

		struct Input {
			std::vector<Connection> _connections;

			bool _state;
			bool _statePrev;

			Input()
				: _state(false), _statePrev(false)
			{}
		};

		struct Node {
			Connection _bias;

			float _activation;

			bool _target;
			bool _state;
			bool _statePrev;

			Node()
				: _target(false), _state(false), _statePrev(false)
			{}
		};

	private:
		int _visibleWidth, _visibleHeight;
		int _hiddenWidth, _hiddenHeight;
		int _visibleToHiddenRadius;
		int _hiddenToVisibleRadiusX;
		int _hiddenToVisibleRadiusY;

		float _hToVx;
		float _hToVy;

		float _vToHx;
		float _vToHy;

		std::vector<BitIndexType> _inputIndices;
		std::vector<BitIndexType> _bitIndices;

		std::vector<Input> _inputs;
		std::vector<Node> _nodes;

	public:
		void create(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int visibleToHiddenRadius, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator);

		void activate(const std::vector<BitIndexType> &inputBitIndices, const std::vector<BitIndexType> &targetBitIndices, float alpha);

		const std::vector<BitIndexType> &getBitIndices() const {
			return _bitIndices;
		}
	};
}