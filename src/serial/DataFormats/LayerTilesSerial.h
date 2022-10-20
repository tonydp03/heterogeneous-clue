#ifndef LayerTiles_h
#define LayerTiles_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "DataFormats/LayerTilesConstants.h"
// #include "DataFormats/cudaCompat.h"
#include "DataFormats/VecArray.h"

class LayerTilesSerial {
public:
  // LayerTilesSerial() { layerTiles_.resize(LayerTilesConstants::nColumns * LayerTilesConstants::nRows); }

  void fill(const std::vector<float>& x, const std::vector<float>& y) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      layerTiles_[getGlobalBin(x[i], y[i])].push_back(i);
    }
  }

  void fill(float x, float y, int i) { layerTiles_[getGlobalBin(x, y)].push_back(i); }

  int getXBin(float x) const {
    constexpr float xRange = LayerTilesConstants::maxX - LayerTilesConstants::minX;
    static_assert(xRange >= 0.);
    int xBin = (x - LayerTilesConstants::minX) * LayerTilesConstants::rX;
    xBin = std::min(xBin, LayerTilesConstants::nColumns - 1);
    xBin = std::max(xBin, 0);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = LayerTilesConstants::maxY - LayerTilesConstants::minY;
    static_assert(yRange >= 0.);
    int yBin = (y - LayerTilesConstants::minY) * LayerTilesConstants::rY;
    yBin = std::min(yBin, LayerTilesConstants::nRows - 1);
    yBin = std::max(yBin, 0);
    return yBin;
  }

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * LayerTilesConstants::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * LayerTilesConstants::nColumns; }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  inline constexpr void clear() {
    for (auto& t : layerTiles_) {
      t.reset();
    }
  }

  inline constexpr void clear(int i) { layerTiles_[i].reset(); }

  // std::vector<int>& operator[](int globalBinId) { return layerTiles_[globalBinId]; }
  cms::cuda::VecArray<int, LayerTilesConstants::maxTileDepth>& operator[](int globalBinId) {
    return layerTiles_[globalBinId];
  }

private:
  // std::vector<std::vector<int>> layerTiles_;
  cms::cuda::VecArray<cms::cuda::VecArray<int, LayerTilesConstants::maxTileDepth>,
                      LayerTilesConstants::nColumns * LayerTilesConstants::nRows>
      layerTiles_;
};

#endif  //LayerTiles_h
