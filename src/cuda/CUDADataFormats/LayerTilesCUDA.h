#ifndef LayerTilesCUDA_h
#define LayerTilesCUDA_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "VecArray.h"
#include "DataFormats/LayerTilesConstants.h"

using CUDAVect = cms::cuda::VecArray<int, LayerTilesConstants::maxTileDepth>;

class LayerTilesCUDA {
public:
  // constructor
  LayerTilesCUDA() = default;

  __device__ void fill(const std::vector<float>& x, const std::vector<float>& y) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      layerTiles_[getGlobalBin(x[i], y[i])].push_back(i);
    }
  }

  __device__ void fill(float x, float y, int i) { layerTiles_[getGlobalBin(x, y)].push_back(i); }

  __host__ __device__ int getXBin(float x) const {
    int xBin = (x - LayerTilesConstants::minX) * LayerTilesConstants::rX;
    xBin = (xBin < LayerTilesConstants::nColumns ? xBin : LayerTilesConstants::nColumns - 1);
    xBin = (xBin > 0 ? xBin : 0);
    return xBin;
  }

  __host__ __device__ int getYBin(float y) const {
    int yBin = (y - LayerTilesConstants::minY) * LayerTilesConstants::rY;
    yBin = (yBin < LayerTilesConstants::nRows ? yBin : LayerTilesConstants::nRows - 1);
    yBin = (yBin > 0 ? yBin : 0);
    ;
    return yBin;
  }
  __device__ int getGlobalBin(float x, float y) const {
    return getXBin(x) + getYBin(y) * LayerTilesConstants::nColumns;
  }

  __host__ __device__ int getGlobalBinByBin(int xBin, int yBin) const {
    return xBin + yBin * LayerTilesConstants::nColumns;
  }

  __host__ __device__ int4 searchBox(float xMin, float xMax, float yMin, float yMax) {
    return int4{getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
  }

  __host__ __device__ void clear() {
    for (auto& t : layerTiles_)
      t.reset();
  }

  __host__ __device__ void clear(int i) {
    layerTiles_[i].reset();
  }

  __host__ __device__ CUDAVect& operator[](int globalBinId) { return layerTiles_[globalBinId]; }

private:
  cms::cuda::VecArray<CUDAVect, LayerTilesConstants::nColumns * LayerTilesConstants::nRows> layerTiles_;
};

#endif
