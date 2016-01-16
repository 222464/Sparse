#include "PredictiveHierarchy.h"

using namespace sparse;

void PredictiveHierarchy::create(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	for (int l = 0; l < _layers.size(); l++) {
		if (l == 0) {
			_layers[l]._sdr.create(_inputWidth, _inputHeight, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._feedForwardRadius, initMinWeight, initMaxWeight, generator);

			_layers[l]._pred.create(_layerDescs[l]._width, _layerDescs[l]._height, _inputWidth, _inputHeight, _layerDescs[l]._predictiveRadius, initMinWeight, initMaxWeight, generator);
		}
		else {
			_layers[l]._sdr.create(_layerDescs[l - 1]._width, _layerDescs[l - 1]._height, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._feedForwardRadius, initMinWeight, initMaxWeight, generator);
		
			_layers[l]._pred.create(_layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l - 1]._width, _layerDescs[l - 1]._height, _layerDescs[l]._predictiveRadius, initMinWeight, initMaxWeight, generator);
		}
	}
}

void PredictiveHierarchy::simStep(const std::vector<BitIndexType> &inputBitIndices) {
	std::vector<BitIndexType> input = inputBitIndices;

	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(input, _layerDescs[l]._inhibitionRadius, _layerDescs[l]._inhibitionStride, _layerDescs[l]._activeCount, _layerDescs[l]._sdrAlpha);

		input = _layers[l]._sdr.getBitIndices();
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == 0)
			_layers[l]._pred.activate(input, inputBitIndices, _layerDescs[l]._predAlpha);
		else
			_layers[l]._pred.activate(input, _layers[l - 1]._sdr.getBitIndices(), _layerDescs[l]._predAlpha);

		input = _layers[l]._pred.getBitIndices();
	}
}