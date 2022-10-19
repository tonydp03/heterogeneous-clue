#ifndef LayerTilesAlpaka_h
#define LayerTilesAlpaka_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaVecArray.h"
#include "DataFormats/LayerTilesConstants.h"

using alpakaVect = cms::alpakatools::VecArray<int, LayerTilesConstants::maxTileDepth>;

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
struct int4 {
  int x, y, z, w;
}; 
#endif

class LayerTilesAlpaka {
public:

  template <typename TAcc>
  ALPAKA_FN_ACC inline constexpr void fill(TAcc& acc, const std::vector<float>& x, const std::vector<float>& y) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      layerTiles_[getGlobalBin(x[i], y[i])].push_back(acc, i);
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline constexpr void fill(TAcc& acc, float x, float y, int i) {
    layerTiles_[getGlobalBin(x, y)].push_back(acc, i);
  }

  ALPAKA_FN_HOST_ACC inline constexpr int getXBin(float x) const {
    int xBin = (x - LayerTilesConstants::minX) * LayerTilesConstants::rX;
    xBin = (xBin < LayerTilesConstants::nColumns ? xBin : LayerTilesConstants::nColumns - 1);
    xBin = (xBin > 0 ? xBin : 0);
    return xBin;
  }

  ALPAKA_FN_HOST_ACC inline constexpr int getYBin(float y) const {
    int yBin = (y - LayerTilesConstants::minY) * LayerTilesConstants::rY;
    yBin = (yBin < LayerTilesConstants::nRows ? yBin : LayerTilesConstants::nRows - 1);
    yBin = (yBin > 0 ? yBin : 0);
    return yBin;
  }
  ALPAKA_FN_HOST_ACC inline constexpr int getGlobalBin(float x, float y) const {
    return getXBin(x) + getYBin(y) * LayerTilesConstants::nColumns;
  }

  ALPAKA_FN_HOST_ACC inline constexpr int getGlobalBinByBin(int xBin, int yBin) const {
    return xBin + yBin * LayerTilesConstants::nColumns;
  }

  ALPAKA_FN_HOST_ACC inline constexpr int4 searchBox(float xMin, float xMax, float yMin, float yMax) {
    return int4{getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
  }

  ALPAKA_FN_HOST_ACC inline constexpr void clear() {
    for (auto& t : layerTiles_)
      t.reset();
  }

  ALPAKA_FN_HOST_ACC inline constexpr void clear(int i) {
    layerTiles_[i].reset();
  }

  ALPAKA_FN_HOST_ACC inline constexpr auto size() {
    return LayerTilesConstants::nColumns * LayerTilesConstants::nRows;
  }



  ALPAKA_FN_HOST_ACC inline constexpr alpakaVect& operator[](int globalBinId) { return layerTiles_[globalBinId]; }

private:
  cms::alpakatools::VecArray<alpakaVect, LayerTilesConstants::nColumns * LayerTilesConstants::nRows> layerTiles_;
};

#endif
