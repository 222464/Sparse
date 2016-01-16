#pragma once

#include "Predictor.h"

namespace sparse {
	class PredictiveHierarchy {
	public:
		struct LayerDesc {
			int _width, _height;

			int _feedForwardRadius, _inhibitionRadius, _recurrentRadius, _predictiveRadius, _feedBackRadius;

			int _inhibitionStride, _activeCount;

			float _sdrAlpha;
			float _predAlpha;

			LayerDesc()
				: _width(16), _height(16),
				_feedForwardRadius(8), _inhibitionRadius(8), _recurrentRadius(8), _predictiveRadius(8), _feedBackRadius(8),
				_inhibitionStride(4), _activeCount(2),
				_sdrAlpha(0.01f), _predAlpha(0.01f)
			{}
		};
		
		struct Layer {
			SDRSPS _sdr;

			Predictor _pred;
		};

	private:
		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

	public:
		void create(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, WeightType initMinWeight, WeightType initMaxWeight, std::mt19937 &generator);

		void simStep(const std::vector<BitIndexType> &inputBitIndices);

		const std::vector<BitIndexType> &getPredBitIndices() const {
			return _layers.front()._pred.getBitIndices();
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}