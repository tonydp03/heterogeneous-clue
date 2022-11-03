// Authors: Marco Rovere, Felice Pantaleo - marco.rovere@cern.ch, felice.pantaleo@cern.ch
// Date: 05/2019

#ifndef TICLLayerTilesAlpaka_h
#define TICLLayerTilesAlpaka_h

#include "DataFormats/Common.h"
#include "DataFormats/Math/normalizedPhi.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaVecArray.h"

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
struct int4 {
  int x, y, z, w;
};
#endif

template <typename T>
class TICLLayerTileT {
public:
  typedef T type;

  template <typename TAcc>
  ALPAKA_FN_ACC void fill(TAcc& acc, double eta, double phi, unsigned int layerClusterId) {
    tiles_[globalBin(eta, phi)].push_back(acc, layerClusterId);
  }

  ALPAKA_FN_HOST_ACC int etaBin(float eta) const {
    constexpr float etaRange = T::maxEta - T::minEta;
    static_assert(etaRange >= 0.f);
    float r = T::nEtaBins / etaRange;
    int etaBin = (std::abs(eta) - T::minEta) * r;
    etaBin = std::clamp(etaBin, 0, T::nEtaBins - 1);
    return etaBin;
  }

  ALPAKA_FN_HOST_ACC int phiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    float r = T::nPhiBins * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;

    return phiBin;
  }

  ALPAKA_FN_HOST_ACC int4 searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
    int etaBinMin = etaBin(etaMin);
    int etaBinMax = etaBin(etaMax);
    int phiBinMin = phiBin(phiMin);
    int phiBinMax = phiBin(phiMax);
    // If the search window cross the phi-bin boundary, add T::nPhiBins to the
    // MAx value. This guarantees that the caller can perform a valid doule
    // loop on eta and phi. It is the caller responsibility to perform a module
    // operation on the phiBin values returned by this function, to explore the
    // correct bins.
    if (phiBinMax < phiBinMin) {
      phiBinMax += T::nPhiBins;
    }
    return int4{etaBinMin, etaBinMax, phiBinMin, phiBinMax};
  }

  ALPAKA_FN_HOST_ACC int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * T::nPhiBins; }

  ALPAKA_FN_HOST_ACC int globalBin(double eta, double phi) const { return phiBin(phi) + etaBin(eta) * T::nPhiBins; }

  // ALPAKA_FN_HOST_ACC void clear() {
  //   auto nBins = T::nEtaBins * T::nPhiBins;
  //   for (int j = 0; j < nBins; ++j)
  //     tiles_[j].clear();
  // }
  ALPAKA_FN_HOST_ACC inline constexpr void clear() {
    for (auto& t : tiles_)
      t.reset();
  }

  ALPAKA_FN_HOST_ACC inline constexpr void clear(int i) { tiles_[i].reset(); }

  ALPAKA_FN_HOST_ACC const cms::alpakatools::VecArray<unsigned int, T::tileDepth>& operator[](int globalBinId) const {
    return tiles_[globalBinId];
  }

private:
  cms::alpakatools::VecArray<cms::alpakatools::VecArray<unsigned int, T::tileDepth>, T::nBins> tiles_;
};

using TICLLayerTilesAlpaka = TICLLayerTileT<ticl::TileConstants>;

#endif
