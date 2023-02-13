#ifndef LayerTilesAlpaka_h
#define LayerTilesAlpaka_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaVecArray.h"
#include "AlpakaVecArrayRef.h"
#include "AlpakaSoAVecArray.h"

#include "DataFormats/LayerTilesConstants.h"

// using alpakaVect = cms::alpakatools::VecArrayRef<int, LayerTilesConstants::maxTileDepth>;

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

  template <typename TAcc>
  ALPAKA_FN_ACC inline constexpr void fill(TAcc& acc, int binId, int i) {
    layerTiles_[binId].push_back(acc, i);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline constexpr void fill_unsafe(TAcc& acc, int binId, int i) {
    layerTiles_[binId].push_back_unsafe(acc, i);
  }


  ALPAKA_FN_HOST_ACC inline constexpr int getXBin(float x) const {
    int xBin = (x - LayerTilesConstants::minX) * LayerTilesConstants::rX;
    xBin = (xBin < LayerTilesConstants::nColumns ? xBin : LayerTilesConstants::nColumns - 1);
    bool xBinPositive = xBin > 0;
    xBin = xBinPositive*xBin;
    return xBin;
  }

  ALPAKA_FN_HOST_ACC inline constexpr int getYBin(float y) const {
    int yBin = (y - LayerTilesConstants::minY) * LayerTilesConstants::rY;
    yBin = (yBin < LayerTilesConstants::nRows ? yBin : LayerTilesConstants::nRows - 1);
    bool yBinPositive = yBin > 0;
    yBin = yBinPositive*yBin;
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
    for(size_t i = 0; i< LayerTilesConstants::nColumns * LayerTilesConstants::nRows; ++i)
      layerTiles_.clear(i);
  }

  ALPAKA_FN_HOST_ACC inline constexpr void clear(int i) {
    layerTiles_.clear(i);
  }

  ALPAKA_FN_HOST_ACC inline constexpr void setPtrs(int i) {
    layerTiles_.setPtrs(i);
  }
  
  ALPAKA_FN_HOST_ACC inline constexpr void init(int i) {
    layerTiles_.init(i);
  }

  ALPAKA_FN_HOST_ACC inline constexpr auto size() {
    return LayerTilesConstants::nColumns * LayerTilesConstants::nRows;
  }



  ALPAKA_FN_HOST_ACC inline constexpr auto& operator[](int globalBinId) { return layerTiles_[globalBinId]; }

private:

  AlpakaSoAVecArray<int, LayerTilesConstants::nColumns * LayerTilesConstants::nRows, LayerTilesConstants::maxTileDepth> layerTiles_;
};

#endif
