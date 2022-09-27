#ifndef CLUE3DAlgo_Serial_Kernels_h
#define CLUE3DAlgo_Serial_Kernels_h

#include "DataFormats/TICLLayerTile.h"
#include "DataFormats/ClusterCollection.h"
#include "DataFormats/Math/deltaR.h"
#include "DataFormats/Math/deltaPhi.h"

// all the functions here need to be changed

void KernelComputeHistogram(TICLLayerTiles &d_hist, ClusterCollectionOnLayers &points) {
//  for (unsigned int i = 0; i < points.n; i++) {
//    // push index of points into tiles
//    d_hist.fill(points.layer[i], points.eta[i], points.phi[i], i);
//  }
};

void KernelCalculateDensity(TICLLayerTiles &d_hist,
    ClusterCollectionOnLayers &points,
    int algoVerbosity = 1,
    int densitySiblingLayers = 3,
    int densityXYDistanceSqr = 3.24,
    float kernelDensityFactor = 0.2,
    bool densityOnSameLayer = false
    ) {

  // To be verified is those numbers are available via types.

  constexpr int nEtaBin = TICLLayerTiles::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TICLLayerTiles::constants_type_t::nPhiBins;
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  for (int layerId = 0; layerId <= nLayers; layerId++) {
    auto &clustersOnLayer = points[layerId];
    unsigned int numberOfClusters = clustersOnLayer.x.size();

    auto isReachable = [](float r0, float r1, float phi0, float phi1, float delta_sqr) -> bool {
      // TODO(rovere): import reco::deltaPhi implementation as well
      auto delta_phi = reco::deltaPhi(phi0, phi1);
      return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi < delta_sqr;
    };
    auto distance_debug = [&](float x1, float x2, float y1, float y2) -> float {
      return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    };

    for (unsigned int i = 0; i < numberOfClusters; i++) {
      // We need to partition the two sides of the HGCAL detector
      int lastLayerPerSide = nLayers / 2;
      int minLayer = 0;
      int maxLayer = 2 * lastLayerPerSide - 1;
      if (layerId < lastLayerPerSide) {
        minLayer = std::max(layerId - densitySiblingLayers, minLayer);
        maxLayer = std::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
      } else {
        minLayer = std::max(layerId - densitySiblingLayers, lastLayerPerSide);
        maxLayer = std::min(layerId + densitySiblingLayers, maxLayer);
      }

      for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
        if (algoVerbosity > 0) {
          std::cout << "RefLayer: " << layerId << " SoaIDX: " << i;
          std::cout << "NextLayer: " << currentLayer;
        }
        const auto &tileOnLayer = d_hist[currentLayer];
        bool onSameLayer = (currentLayer == layerId);
        if (algoVerbosity > 0) {
          std::cout << "onSameLayer: " << onSameLayer;
        }
        const int etaWindow = 2;
        const int phiWindow = 2;
        int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
        int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
        int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
        int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
        if (algoVerbosity > 0) {
          std::cout << "eta: " << clustersOnLayer.eta[i];
          std::cout << "phi: " << clustersOnLayer.phi[i];
          std::cout << "etaBinMin: " << etaBinMin << ", etaBinMax: " << etaBinMax;
          std::cout << "phiBinMin: " << phiBinMin << ", phiBinMax: " << phiBinMax;
        }
        for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
          auto offset = ieta * nPhiBin;
          if (algoVerbosity > 0) {
            std::cout << "offset: " << offset;
          }
          for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
            int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
            if (algoVerbosity > 0) {
              std::cout << "iphi: " << iphi;
              std::cout
                << "Entries in tileBin: " << tileOnLayer[offset + iphi].size();
            }
            for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
              auto const &clustersLayer = points[currentLayer];
              if (algoVerbosity > 0) {
                std::cout
                  << "OtherLayer: " << currentLayer << " SoaIDX: " << otherClusterIdx;
                std::cout << "OtherEta: " << clustersLayer.eta[otherClusterIdx];
                std::cout << "OtherPhi: " << clustersLayer.phi[otherClusterIdx];
              }
              bool reachable = false;
              // Still differentiate between silicon and Scintillator.
              // Silicon has yet to be studied further.
              if (clustersOnLayer.isSilicon[i]) {
                reachable = isReachable(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                    clustersLayer.r_over_absz[otherClusterIdx] * clustersOnLayer.z[i],
                    clustersOnLayer.phi[i],
                    clustersLayer.phi[otherClusterIdx],
                    densityXYDistanceSqr);
              } else {
                reachable = isReachable(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                    clustersLayer.r_over_absz[otherClusterIdx] * clustersOnLayer.z[i],
                    clustersOnLayer.phi[i],
                    clustersLayer.phi[otherClusterIdx],
                    clustersOnLayer.radius[i] * clustersOnLayer.radius[i]);
              }
              if (algoVerbosity > 0) {
                std::cout << "Distance[eta,phi]: "
                  << reco::deltaR2(clustersOnLayer.eta[i],
                      clustersOnLayer.phi[i],
                      clustersLayer.eta[otherClusterIdx],
                      clustersLayer.phi[otherClusterIdx]);
                auto dist = distance_debug(
                    clustersOnLayer.r_over_absz[i],
                    clustersLayer.r_over_absz[otherClusterIdx],
                    clustersOnLayer.r_over_absz[i] * std::abs(clustersOnLayer.phi[i]),
                    clustersLayer.r_over_absz[otherClusterIdx] * std::abs(clustersLayer.phi[otherClusterIdx]));
                std::cout << "Distance[cm]: " << (dist * clustersOnLayer.z[i]);
                std::cout
                  << "Energy Other:   " << clustersLayer.energy[otherClusterIdx];
                std::cout << "Cluster radius: " << clustersOnLayer.radius[i];
              }
              if (reachable) {
                float factor_same_layer_different_cluster = (onSameLayer && !densityOnSameLayer) ? 0.f : 1.f;
                auto energyToAdd = (onSameLayer && (i == otherClusterIdx)
                    ? 1.f
                    : kernelDensityFactor * factor_same_layer_different_cluster) *
                  clustersLayer.energy[otherClusterIdx];
                clustersOnLayer.rho[i] += energyToAdd;
                if (algoVerbosity > 0) {
                  std::cout
                    << "Adding " << energyToAdd << " partial " << clustersOnLayer.rho[i];
                }
              }
            }  // end of loop on possible compatible clusters
          }    // end of loop over phi-bin region
        }      // end of loop over eta-bin region
      }        // end of loop on the sibling layers
    }  // end of loop over clusters on this layer
  }
}

void KernelComputeDistanceToHigher(TICLLayerTiles &d_hist,
                                   ClusterCollectionOnLayers &points,
                                   float outlierDeltaFactor,
                                   float dc) {
  return;
};

void KernelFindAndAssignClusters(ClusterCollectionOnLayers &points, float outlierDeltaFactor, float dc, float rhoc) {
  return;
};

#endif
