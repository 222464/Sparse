#include "SDRSPS.h"

using namespace sparse;

void SDRSPS::create(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int visibleToHiddenRadius, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator) {
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

		_nodes[hi]._activation = _nodes[hi]._bias._weight;
	}
}

void SDRSPS::activate(const std::vector<BitIndexType> &inputBitIndices, int inhibitionSize, int inhibitionStride, int activeCount, float alpha) {
	// Reset previous activations
	for (int i = 0; i < _bitIndices.size(); i++) {
		int hi = _bitIndices[i];

		_nodes[hi]._activation = _nodes[hi]._bias._weight;

		_nodes[hi]._state = false;
	}

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
				}

				ci++;
			}
	}

	std::unordered_set<int> newBitIndices;

	auto cmp = [](std::pair<float, int> left, std::pair<float, int> right) { return left.first > right.first; };

	int mainArea = inhibitionSize * inhibitionSize;

	// Group into chunks and inhibit via a sorting algorithm
	for (int cx = 0; cx < _hiddenWidth; cx += inhibitionStride)
		for (int cy = 0; cy < _hiddenHeight; cy += inhibitionStride) {		
			std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> data(cmp);

			int area = std::min(inhibitionSize, _hiddenWidth - cx) * std::min(inhibitionSize, _hiddenHeight - cy);

			float areaRatio = static_cast<float>(area) / mainArea;

			int active = std::round(areaRatio * activeCount);

			for (int dx = 0; dx < std::min(inhibitionSize, _hiddenWidth - cx); dx++)
				for (int dy = 0; dy < std::min(inhibitionSize, _hiddenHeight - cy); dy++) {
					int x = cx + dx;
					int y = cy + dy;

					int hio = x + y * _hiddenWidth;

					data.push(std::pair<float, int>(_nodes[hio]._activation, hio));
				}

			// Set top N
			for (int n = 0; n < active; n++) {
				int ti = data.top().second;

				data.pop();

				_nodes[ti]._state = true;

				if (newBitIndices.find(ti) == newBitIndices.end())
					newBitIndices.insert(ti);
			}
		}

	// Create vector from set
	_bitIndices.resize(newBitIndices.size());

	int index = 0;

	for (std::unordered_set<int>::const_iterator cit = newBitIndices.begin(); cit != newBitIndices.end(); cit++)
		_bitIndices[index++] = *cit;

	// Learn
	const int visibleToHiddenDiam = _visibleToHiddenRadius * 2 + 1;

	for (int i = 0; i < _bitIndices.size(); i++) {
		int hi = _bitIndices[i];

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

						_inputs[vi]._connections[ci]._weight += alpha * _nodes[hi]._activation * ((_inputs[vi]._state ? 1.0f : 0.0f) - _nodes[hi]._activation * _inputs[vi]._connections[ci]._weight);
					}
				}
			}
	}

	// Unset inputs
	for (int i = 0; i < inputBitIndices.size(); i++)
		_inputs[inputBitIndices[i]]._state = false;
}