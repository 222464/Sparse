#pragma once

#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <queue>

namespace sparse {
	typedef int BitIndexType;
	typedef float WeightType;

	class SDRSPS {
	public:
		struct Connection {
			WeightType _weight;
		};

		struct Input {
			std::vector<Connection> _connections;

			bool _state;

			Input()
				: _state(false)
			{}
		};

		struct Node {
			Connection _bias;
	
			float _activation;

			bool _state;

			Node()
				: _state(false)
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

		std::vector<BitIndexType> _bitIndices;
	
		std::vector<Input> _inputs;
		std::vector<Node> _nodes;

	public:
		void create(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int visibleToHiddenRadius, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator);

		void activate(const std::vector<BitIndexType> &inputBitIndices, int inhibitionSize, int inhibitionStride, int activeCount, float alpha);

		const std::vector<BitIndexType> &getBitIndices() const {
			return _bitIndices;
		}
	};
}