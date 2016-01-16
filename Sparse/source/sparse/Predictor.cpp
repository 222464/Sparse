#include "Predictor.h"
#include <iostream>
using namespace sparse;

void Predictor::create(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int visibleToHiddenRadius, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator) {
	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;
	_visibleToHiddenRadius = visibleToHiddenRadius;

	std::uniform_real_distribution<WeightType> weightDist(initMinWeight, initMaxWeight);

	_inputs.clear();
	_inputs.resize(_visibleWidth * _visibleHeight);

	_nodes.clear();
	_nodes.resize(_hiddenWidth * _hiddenHeight);

	_hToVx = static_cast<float>(_visibleWidth) / _hiddenWidth;
	_hToVy = static_cast<float>(_visibleHeight) / _hiddenHeight;

	_vToHx = 1.0f / _hToVx;
	_vToHy = 1.0f / _hToVy;

	_hiddenToVisibleRadiusX = std::ceil(_vToHx * _visibleToHiddenRadius);
	_hiddenToVisibleRadiusY = std::ceil(_vToHy * _visibleToHiddenRadius);

	const int visibleToHiddenDiam = _visibleToHiddenRadius * 2 + 1;
	const int visibleToHiddenNumConnections = visibleToHiddenDiam * visibleToHiddenDiam;

	// Init inputs
	for (int vx = 0; vx < _visibleWidth; vx++)
		for (int vy = 0; vy < _visibleHeight; vy++) {
			int vi = vx + vy * _visibleWidth;

			_inputs[vi]._connections.resize(visibleToHiddenNumConnections);

			for (int ci = 0; ci < visibleToHiddenNumConnections; ci++)
				_inputs[vi]._connections[ci]._weight = weightDist(generator);
		}

	// Init nodes
	for (int hi = 0; hi < _nodes.size(); hi++) {
		_nodes[hi]._bias._weight = weightDist(generator);

		_nodes[hi]._activation = 0.0f;// _nodes[hi]._bias._weight;
	}
}

void Predictor::activate(const std::vector<BitIndexType> &inputBitIndices, const std::vector<BitIndexType> &targetBitIndices, float alpha) {
	std::vector<BitIndexType> bitIndicesPrev = _bitIndices;
	std::vector<BitIndexType> inputIndicesPrev = _inputIndices;

	// Set input previous states
	for (int i = 0; i < _inputIndices.size(); i++)
		_inputs[_inputIndices[i]]._statePrev = true;

	// Set target states
	for (int i = 0; i < targetBitIndices.size(); i++)
		_nodes[targetBitIndices[i]]._target = true;

	// Reset previous activations
	for (int i = 0; i < _bitIndices.size(); i++) {
		int hi = _bitIndices[i];

		_nodes[hi]._activation = 0.0f;// _nodes[hi]._bias._weight;

		_nodes[hi]._state = false;

		_nodes[hi]._statePrev = true;
	}

	// Learn
	const int visibleToHiddenDiam = _visibleToHiddenRadius * 2 + 1;

	// Find disjoint bits
	std::unordered_set<int> targetBitSet;
	std::unordered_set<int> predBitSet;

	for (int i = 0; i < targetBitIndices.size(); i++)
		targetBitSet.insert(targetBitIndices[i]);

	for (int i = 0; i < _bitIndices.size(); i++)
		predBitSet.insert(_bitIndices[i]);

	std::vector<BitIndexType> disjointBits;

	for (int i = 0; i < targetBitIndices.size(); i++)
		if (predBitSet.find(targetBitIndices[i]) == predBitSet.end())
			disjointBits.push_back(targetBitIndices[i]);

	for (int i = 0; i < _bitIndices.size(); i++)
		if (targetBitSet.find(_bitIndices[i]) == targetBitSet.end())
			disjointBits.push_back(_bitIndices[i]);

	for (int i = 0; i < disjointBits.size(); i++) {
		int hi = disjointBits[i];

		int hx = hi % _hiddenWidth;
		int hy = hi / _hiddenWidth;

		// Go through connections
		int vCenterX = std::round(_hToVx * hx);
		int vCenterY = std::round(_hToVy * hy);

		for (int dx = -_hiddenToVisibleRadiusX; dx <= _hiddenToVisibleRadiusX; dx++)
			for (int dy = -_hiddenToVisibleRadiusY; dy <= _hiddenToVisibleRadiusY; dy++) {
				int vx = vCenterX + dx;
				int vy = vCenterY + dy;

				if (vx >= 0 && vy >= 0 && vx < _visibleWidth && vy < _visibleHeight) {
					int vi = vx + vy * _visibleWidth;

					int hCenterX = std::round(_vToHx * vx);
					int hCenterY = std::round(_vToHy * vy);

					int minX = hCenterX - _visibleToHiddenRadius;
					int minY = hCenterY - _visibleToHiddenRadius;

					int maxX = hCenterX + _visibleToHiddenRadius;
					int maxY = hCenterY + _visibleToHiddenRadius;

					if (hx >= minX && hy >= minY && hx <= maxX && hy <= maxY) {
						// This node is being addressed
						int odx = hx - minX;
						int ody = hy - minY;

						int ci = ody + odx * visibleToHiddenDiam;

						_inputs[vi]._connections[ci]._weight += alpha * ((_nodes[hi]._target ? 1.0f : 0.0f) - (_nodes[hi]._statePrev ? 1.0f : 0.0f)) * (_inputs[vi]._statePrev ? 1.0f : 0.0f);
					}
				}
			}
	}

	std::unordered_set<int> activeIndices;

	// Go through input indices and activate nodes
	for (int i = 0; i < inputBitIndices.size(); i++) {
		int vi = inputBitIndices[i];

		_inputs[vi]._state = true;

		int vx = vi % _visibleWidth;
		int vy = vi / _visibleWidth;

		// Find center of references
		int centerX = std::round(vx * _vToHx);
		int centerY = std::round(vy * _vToHy);

		int ci = 0;

		for (int dx = -_visibleToHiddenRadius; dx <= _visibleToHiddenRadius; dx++)
			for (int dy = -_visibleToHiddenRadius; dy <= _visibleToHiddenRadius; dy++) {
				int hx = centerX + dx;
				int hy = centerY + dy;

				if (hx >= 0 && hy >= 0 && hx < _hiddenWidth && hy < _hiddenHeight) {
					int hi = hx + hy * _hiddenWidth;

					_nodes[hi]._activation += _inputs[vi]._connections[ci]._weight;

					if (_nodes[hi]._activation > 0.5f) {
						if (activeIndices.find(hi) == activeIndices.end())
							activeIndices.insert(hi);
					}
					else {
						if (activeIndices.find(hi) != activeIndices.end())
							activeIndices.erase(hi);
					}
				}

				ci++;
			}
	}

	// Create vector from set
	_bitIndices.resize(activeIndices.size());

	int index = 0;

	for (std::unordered_set<int>::const_iterator cit = activeIndices.begin(); cit != activeIndices.end(); cit++)
		_bitIndices[index++] = *cit;

	// Unset inputs
	for (int i = 0; i < inputBitIndices.size(); i++)
		_inputs[inputBitIndices[i]]._state = false;

	// Uset previous inputs
	for (int i = 0; i < inputIndicesPrev.size(); i++)
		_inputs[inputIndicesPrev[i]]._statePrev = false;

	// Unset target states
	for (int i = 0; i < targetBitIndices.size(); i++)
		_nodes[targetBitIndices[i]]._target = false;

	// Reset previous states
	for (int i = 0; i < bitIndicesPrev.size(); i++) {
		int hi = bitIndicesPrev[i];

		_nodes[hi]._statePrev = false;
	}

	_inputIndices = inputBitIndices;
}