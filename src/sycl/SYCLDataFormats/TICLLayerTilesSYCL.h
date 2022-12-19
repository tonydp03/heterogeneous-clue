// Authors: Marco Rovere, Felice Pantaleo - marco.rovere@cern.ch, felice.pantaleo@cern.ch
// Date: 05/2019

#ifndef TICLLayerTilesSYCL_h
#define TICLLayerTilesSYCL_h

#include <CL/sycl.hpp>

#include "DataFormats/Common.h"
#include "DataFormats/Math/normalizedPhi.h"
#include "SYCLCore/VecArray.h"

template <typename T>
class TICLLayerTileT {
public:
  typedef T type;
  void fill(double eta, double phi, unsigned int layerClusterId) {
    tiles_[globalBin(eta, phi)].push_back(layerClusterId);
  }

  int etaBin(float eta) const {
    constexpr float etaRange = T::maxEta - T::minEta;
    static_assert(etaRange >= 0.f);
    float r = T::nEtaBins / etaRange;
    int etaBin = (sycl::abs(eta) - T::minEta) * r;
    etaBin = std::clamp(etaBin, 0, T::nEtaBins - 1);
    return etaBin;
  }

  int phiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    float r = T::nPhiBins * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;

    return phiBin;
  }

  sycl::int4 searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
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
    return sycl::int4{etaBinMin, etaBinMax, phiBinMin, phiBinMax};
  }

  int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * T::nPhiBins; }

  int globalBin(double eta, double phi) const { return phiBin(phi) + etaBin(eta) * T::nPhiBins; }

  inline constexpr void clear() {
    for (auto& t : tiles_)
      t.reset();
  }

  inline constexpr void clear(int i) { tiles_[i].reset(); }

  const cms::sycltools::VecArray<unsigned int, T::tileDepth>& operator[](int globalBinId) const {
    return tiles_[globalBinId];
  }

private:
  cms::sycltools::VecArray<cms::sycltools::VecArray<unsigned int, T::tileDepth>, T::nBins> tiles_;
};

using TICLLayerTilesSYCL = TICLLayerTileT<ticl::TileConstants>;

#endif
