#ifndef CLUE3DAlgo_Serial_Kernels_h
#define CLUE3DAlgo_Serial_Kernels_h

#include "DataFormats/TICLLayerTile.h"
#include "DataFormats/ClusterCollection.h"
#include "DataFormats/Math/deltaR.h"
#include "DataFormats/Math/deltaPhi.h"

// all the functions here need to be changed

void KernelComputeHistogram(TICLLayerTiles &d_hist, ClusterCollectionSerialOnLayers &points) {
  for (unsigned int layer = 0; layer < points.size(); layer++) {
    for (unsigned int idxSoAOnLyr = 0; idxSoAOnLyr < points[layer].x.size(); ++idxSoAOnLyr) {
      d_hist.fill(layer, std::abs(points[layer].eta[idxSoAOnLyr]), points[layer].phi[idxSoAOnLyr], idxSoAOnLyr);
    }
  }
};

void KernelComputeHistogramSoA(TICLLayerTiles &d_histSoA, ClusterCollectionSerial &points) {
  for (unsigned int clusterIdxSoA = 0; clusterIdxSoA < points.x.size(); clusterIdxSoA++) {
    d_histSoA.fill(
        points.layer[clusterIdxSoA], std::abs(points.eta[clusterIdxSoA]), points.phi[clusterIdxSoA], clusterIdxSoA);
  }
};

void KernelCalculateDensitySoA(TICLLayerTiles &d_hist,
                               ClusterCollectionSerial &points,
                               int algoVerbosity = 0,
                               int densitySiblingLayers = 3,
                               int densityXYDistanceSqr = 3.24,
                               float kernelDensityFactor = 0.2,
                               bool densityOnSameLayer = false) {
  constexpr int nEtaBin = TICLLayerTiles::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TICLLayerTiles::constants_type_t::nPhiBins;
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  for (unsigned int clusterIdxSoA = 0; clusterIdxSoA < points.x.size(); ++clusterIdxSoA) {
    int layerId = points.layer[clusterIdxSoA];

    auto isReachable = [](float r0, float r1, float phi0, float phi1, float delta_sqr) -> bool {
      // TODO(rovere): import reco::deltaPhi implementation as well
      auto delta_phi = reco::deltaPhi(phi0, phi1);
      return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi < delta_sqr;
    };
    auto distance_debug = [&](float x1, float x2, float y1, float y2) -> float {
      return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    };

    // We need to partition the two sides of the HGCAL detector
    constexpr int lastLayerPerSide = nLayers / 2;
    int maxLayer = 2 * lastLayerPerSide - 1;
    int minLayer = 0;
    if (layerId < lastLayerPerSide) {
      minLayer = std::max(layerId - densitySiblingLayers, minLayer);
      maxLayer = std::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
    } else {
      minLayer = std::max(layerId - densitySiblingLayers, lastLayerPerSide);
      maxLayer = std::min(layerId + densitySiblingLayers, maxLayer);
    }

    for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      if (algoVerbosity > 0) {
        std::cout << "RefLayer: " << layerId << " SoaIDX: " << clusterIdxSoA;
        std::cout << "NextLayer: " << currentLayer;
      }
      const auto &tileOnLayer = d_hist[currentLayer];
      bool onSameLayer = (currentLayer == layerId);
      if (algoVerbosity > 0) {
        std::cout << "onSameLayer: " << onSameLayer;
      }
      const int etaWindow = 2;
      const int phiWindow = 2;
      int etaBinMin = std::max(tileOnLayer.etaBin(points.eta[clusterIdxSoA]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(points.eta[clusterIdxSoA]) + etaWindow, nEtaBin);
      int phiBinMin = tileOnLayer.phiBin(points.phi[clusterIdxSoA]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(points.phi[clusterIdxSoA]) + phiWindow;
      if (algoVerbosity > 0) {
        std::cout << "eta: " << points.eta[clusterIdxSoA] << std::endl;
        std::cout << "phi: " << points.phi[clusterIdxSoA] << std::endl;
        std::cout << "etaBinMin: " << etaBinMin << ", etaBinMax: " << etaBinMax << std::endl;
        std::cout << "phiBinMin: " << phiBinMin << ", phiBinMax: " << phiBinMax << std::endl;
      }
      for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        if (algoVerbosity > 0) {
          std::cout << "offset: " << offset << std::endl;
        }
        for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          if (algoVerbosity > 0) {
            std::cout << "iphi: " << iphi << std::endl;
            std::cout << "Entries in tileBin: " << tileOnLayer[offset + iphi].size() << std::endl;
          }
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            //              auto const &clustersLayer = points[currentLayer];
            if (algoVerbosity > 0) {
              std::cout << "OtherLayer: " << currentLayer << " SoaIDX: " << otherClusterIdx << std::endl;
              std::cout << "OtherEta: " << points.eta[otherClusterIdx] << std::endl;
              std::cout << "OtherPhi: " << points.phi[otherClusterIdx] << std::endl;
            }
            bool reachable = false;
            // Still differentiate between silicon and Scintillator.
            // Silicon has yet to be studied further.
            if (points.isSilicon[clusterIdxSoA]) {
              reachable = isReachable(points.r_over_absz[clusterIdxSoA] * points.z[clusterIdxSoA],
                                      points.r_over_absz[otherClusterIdx] * points.z[clusterIdxSoA],
                                      points.phi[clusterIdxSoA],
                                      points.phi[otherClusterIdx],
                                      densityXYDistanceSqr);
            } else {
              reachable = isReachable(points.r_over_absz[clusterIdxSoA] * points.z[clusterIdxSoA],
                                      points.r_over_absz[otherClusterIdx] * points.z[clusterIdxSoA],
                                      points.phi[clusterIdxSoA],
                                      points.phi[otherClusterIdx],
                                      points.radius[clusterIdxSoA] * points.radius[clusterIdxSoA]);
            }
            if (algoVerbosity > 0) {
              std::cout << "Distance[eta,phi]: "
                        << reco::deltaR2(points.eta[clusterIdxSoA],
                                         points.phi[clusterIdxSoA],
                                         points.eta[otherClusterIdx],
                                         points.phi[otherClusterIdx])
                        << std::endl;
              auto dist = distance_debug(points.r_over_absz[clusterIdxSoA],
                                         points.r_over_absz[otherClusterIdx],
                                         points.r_over_absz[clusterIdxSoA] * std::abs(points.phi[clusterIdxSoA]),
                                         points.r_over_absz[otherClusterIdx] * std::abs(points.phi[otherClusterIdx]));
              std::cout << "Distance[cm]: " << (dist * points.z[clusterIdxSoA]) << std::endl;
              std::cout << "Energy Other:   " << points.energy[otherClusterIdx] << std::endl;
              std::cout << "Cluster radius: " << points.radius[clusterIdxSoA] << std::endl;
            }
            if (reachable) {
              float factor_same_layer_different_cluster = (onSameLayer && !densityOnSameLayer) ? 0.f : 1.f;
              auto energyToAdd =
                  ((clusterIdxSoA == otherClusterIdx) ? 1.f
                                                      : kernelDensityFactor * factor_same_layer_different_cluster) *
                  points.energy[otherClusterIdx];
              points.rho[clusterIdxSoA] += energyToAdd;
              if (algoVerbosity > 0) {
                std::cout << "Adding " << energyToAdd << " partial " << points.rho[clusterIdxSoA] << std::endl;
              }
            }
          }  // end of loop on possible compatible clusters
        }    // end of loop over phi-bin region
      }      // end of loop over eta-bin region
    }        // end of loop on the sibling layers
  }
  //  for (unsigned int i = 0; i < points.rho.size(); ++i) {
  //    std::cout << "Layer " << points.layer[i] << " i " << i << " rho " << points.rho[i] << std::endl;
  //  }
}

void KernelCalculateDensity(TICLLayerTiles &d_hist,
                            ClusterCollectionSerialOnLayers &points,
                            int algoVerbosity = 0,
                            int densitySiblingLayers = 3,
                            int densityXYDistanceSqr = 3.24,
                            float kernelDensityFactor = 0.2,
                            bool densityOnSameLayer = false) {
  // To be verified is those numbers are available via types.

  constexpr int nEtaBin = TICLLayerTiles::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TICLLayerTiles::constants_type_t::nPhiBins;
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  for (int layerId = 0; layerId < nLayers; layerId++) {
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
          std::cout << "RefLayer: " << layerId << " SoaIDX: " << i << std::endl;
          std::cout << "NextLayer: " << currentLayer << std::endl;
        }
        const auto &tileOnLayer = d_hist[currentLayer];
        bool onSameLayer = (currentLayer == layerId);
        if (algoVerbosity > 0) {
          std::cout << "onSameLayer: " << onSameLayer << std::endl;
        }
        const int etaWindow = 2;
        const int phiWindow = 2;
        int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
        int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
        int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
        int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
        if (algoVerbosity > 0) {
          std::cout << "eta: " << clustersOnLayer.eta[i] << std::endl;
          std::cout << "phi: " << clustersOnLayer.phi[i] << std::endl;
          std::cout << "etaBinMin: " << etaBinMin << ", etaBinMax: " << etaBinMax << std::endl;
          std::cout << "phiBinMin: " << phiBinMin << ", phiBinMax: " << phiBinMax << std::endl;
        }
        for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
          auto offset = ieta * nPhiBin;
          if (algoVerbosity > 0) {
            std::cout << "offset: " << offset << std::endl;
          }
          for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
            int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
            if (algoVerbosity > 0) {
              std::cout << "iphi: " << iphi << std::endl;
              std::cout << "Entries in tileBin: " << tileOnLayer[offset + iphi].size() << std::endl;
            }
            for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
              auto const &clustersLayer = points[currentLayer];
              if (algoVerbosity > 0) {
                std::cout << "OtherLayer: " << currentLayer << " SoaIDX: " << otherClusterIdx << std::endl;
                std::cout << "OtherEta: " << clustersLayer.eta[otherClusterIdx] << std::endl;
                std::cout << "OtherPhi: " << clustersLayer.phi[otherClusterIdx] << std::endl;
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
                                           clustersLayer.phi[otherClusterIdx])
                          << std::endl;
                auto dist = distance_debug(
                    clustersOnLayer.r_over_absz[i],
                    clustersLayer.r_over_absz[otherClusterIdx],
                    clustersOnLayer.r_over_absz[i] * std::abs(clustersOnLayer.phi[i]),
                    clustersLayer.r_over_absz[otherClusterIdx] * std::abs(clustersLayer.phi[otherClusterIdx]));
                std::cout << "Distance[cm]: " << (dist * clustersOnLayer.z[i]) << std::endl;
                std::cout << "Energy Other:   " << clustersLayer.energy[otherClusterIdx] << std::endl;
                std::cout << "Cluster radius: " << clustersOnLayer.radius[i] << std::endl;
              }
              if (reachable) {
                float factor_same_layer_different_cluster = (onSameLayer && !densityOnSameLayer) ? 0.f : 1.f;
                auto energyToAdd = (onSameLayer && (i == otherClusterIdx)
                                        ? 1.f
                                        : kernelDensityFactor * factor_same_layer_different_cluster) *
                                   clustersLayer.energy[otherClusterIdx];
                clustersOnLayer.rho[i] += energyToAdd;
                if (algoVerbosity > 0) {
                  std::cout << "Adding " << energyToAdd << " partial " << clustersOnLayer.rho[i] << std::endl;
                }
              }
            }  // end of loop on possible compatible clusters
          }    // end of loop over phi-bin region
        }      // end of loop over eta-bin region
      }        // end of loop on the sibling layers
    }          // end of loop over clusters on this layer
  }
  //  for (unsigned int l = 0; l < nLayers; ++l) {
  //    for (unsigned int i = 0; i < points[l].rho.size(); ++i) {
  //      std::cout << "Layer:" << l << " i " << i << " rho " << points[l].rho[i] << std::endl;
  //    }
  //  }
}

void KernelComputeDistanceToHigherSoA(TICLLayerTiles &d_hist,
                                      ClusterCollectionSerial &points,
                                      int algoVerbosity = 0,
                                      int densitySiblingLayers = 3,
                                      bool nearestHigherOnSameLayer = false) {
  constexpr int nEtaBin = TICLLayerTiles::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TICLLayerTiles::constants_type_t::nPhiBins;
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  for (unsigned int clusterIdxSoA = 0; clusterIdxSoA < points.x.size(); ++clusterIdxSoA) {
    int layerId = points.layer[clusterIdxSoA];

    auto distanceSqr = [](float r0, float r1, float phi0, float phi1) -> float {
      auto delta_phi = reco::deltaPhi(phi0, phi1);
      return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi;
    };

    if (algoVerbosity > 0) {
      std::cout << "Starting searching nearestHigher on " << layerId << " with rho: " << points.rho[clusterIdxSoA]
                << " at eta, phi: " << d_hist[layerId].etaBin(points.eta[clusterIdxSoA]) << ", "
                << d_hist[layerId].phiBin(points.phi[clusterIdxSoA]);
    }
    // We need to partition the two sides of the HGCAL detector
    constexpr int lastLayerPerSide = nLayers / 2;
    int minLayer = 0;
    int maxLayer = 2 * lastLayerPerSide - 1;
    if (layerId < lastLayerPerSide) {
      minLayer = std::max(layerId - densitySiblingLayers, minLayer);
      maxLayer = std::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
    } else {
      minLayer = std::max(layerId - densitySiblingLayers, lastLayerPerSide + 1);
      maxLayer = std::min(layerId + densitySiblingLayers, maxLayer);
    }
    constexpr float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    std::pair<int, int> i_nearestHigher(-1, -1);
    std::pair<float, int> nearest_distances(maxDelta, std::numeric_limits<int>::max());
    for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      if (!nearestHigherOnSameLayer && (layerId == currentLayer))
        continue;
      const auto &tileOnLayer = d_hist[currentLayer];
      int etaWindow = 1;
      int phiWindow = 1;
      int etaBinMin = std::max(tileOnLayer.etaBin(points.eta[clusterIdxSoA]) - etaWindow, 0);
      int etaBinMax = std::min(tileOnLayer.etaBin(points.eta[clusterIdxSoA]) + etaWindow, nEtaBin);
      int phiBinMin = tileOnLayer.phiBin(points.phi[clusterIdxSoA]) - phiWindow;
      int phiBinMax = tileOnLayer.phiBin(points.phi[clusterIdxSoA]) + phiWindow;
      for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
        auto offset = ieta * nPhiBin;
        for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
          int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
          if (algoVerbosity > 0) {
            std::cout << "Searching nearestHigher on " << currentLayer << " eta, phi: " << ieta << ", " << iphi_it
                      << " " << iphi << " " << offset << " " << (offset + iphi);
          }
          for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
            //              auto const &clustersOnOtherLayer = points[currentLayer];
            auto dist = maxDelta;
            auto dist_transverse = maxDelta;
            int dist_layers = std::abs(currentLayer - layerId);
            dist_transverse = distanceSqr(points.r_over_absz[clusterIdxSoA] * points.z[clusterIdxSoA],
                                          points.r_over_absz[otherClusterIdx] * points.z[clusterIdxSoA],
                                          points.phi[clusterIdxSoA],
                                          points.phi[otherClusterIdx]);
            // Add Z-scale to the final distance
            dist = dist_transverse;
            // TODO(rovere): in case of equal local density, the ordering in
            // the original CLUE3D implementaiton is bsaed on the index of
            // the LayerCclusters in the LayerClusterCollection. In this
            // case, the index is based on the ordering of the SOA indices.
            // bool foundHigher =
            //     (points.rho[otherClusterIdx] > points.rho[clusterIdxSoA]) ||
            //     (points.rho[otherClusterIdx] == points.rho[clusterIdxSoA] && otherClusterIdx > clusterIdxSoA);
            bool foundHigher = (points.rho[otherClusterIdx] > points.rho[clusterIdxSoA]);
            if (algoVerbosity > 0) {
              std::cout << "Searching nearestHigher on " << currentLayer << " with rho: " << points.rho[otherClusterIdx]
                        << " on layerIdxInSOA: " << currentLayer << ", " << otherClusterIdx
                        << " with distance: " << sqrt(dist) << " foundHigher: " << foundHigher;
            }
            if (foundHigher && dist <= i_delta) {
              // update i_delta
              i_delta = dist;
              nearest_distances = std::make_pair(sqrt(dist_transverse), dist_layers);
              // update i_nearestHigher
              i_nearestHigher = std::make_pair(currentLayer, otherClusterIdx);
            }
          }  // End of loop on clusters
        }    // End of loop on phi bins
      }      // End of loop on eta bins
    }        // End of loop on layers

    bool foundNearestInFiducialVolume = (i_delta != maxDelta);
    if (algoVerbosity > 0) {
      std::cout << "i_delta: " << i_delta << " passed: " << foundNearestInFiducialVolume << " " << i_nearestHigher.first
                << " " << i_nearestHigher.second << " distances: " << nearest_distances.first << ", "
                << nearest_distances.second;
    }
    if (foundNearestInFiducialVolume) {
      points.delta[clusterIdxSoA] = nearest_distances;
      points.nearestHigher[clusterIdxSoA] = i_nearestHigher;
    } else {
      // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
      // we can safely maximize delta to be maxDelta
      points.delta[clusterIdxSoA] = std::make_pair(maxDelta, std::numeric_limits<int>::max());
      points.nearestHigher[clusterIdxSoA] = {-1, -1};
    }
  }
};

void KernelComputeDistanceToHigher(TICLLayerTiles &d_hist,
                                   ClusterCollectionSerialOnLayers &points,
                                   int algoVerbosity = 0,
                                   int densitySiblingLayers = 3,
                                   bool nearestHigherOnSameLayer = false) {
  constexpr int nEtaBin = TICLLayerTiles::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TICLLayerTiles::constants_type_t::nPhiBins;
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  for (int layerId = 0; layerId < nLayers; layerId++) {
    auto &clustersOnLayer = points[layerId];
    unsigned int numberOfClusters = clustersOnLayer.x.size();

    auto distanceSqr = [](float r0, float r1, float phi0, float phi1) -> float {
      auto delta_phi = reco::deltaPhi(phi0, phi1);
      return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi;
    };

    for (unsigned int i = 0; i < numberOfClusters; i++) {
      if (algoVerbosity > 0) {
        std::cout << "Starting searching nearestHigher on " << layerId << " with rho: " << clustersOnLayer.rho[i]
                  << " at eta, phi: " << d_hist[layerId].etaBin(clustersOnLayer.eta[i]) << ", "
                  << d_hist[layerId].phiBin(clustersOnLayer.phi[i]);
      }
      // We need to partition the two sides of the HGCAL detector
      int lastLayerPerSide = nLayers / 2;
      int minLayer = 0;
      int maxLayer = 2 * lastLayerPerSide - 1;
      if (layerId < lastLayerPerSide) {
        minLayer = std::max(layerId - densitySiblingLayers, minLayer);
        maxLayer = std::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
      } else {
        minLayer = std::max(layerId - densitySiblingLayers, lastLayerPerSide + 1);
        maxLayer = std::min(layerId + densitySiblingLayers, maxLayer);
      }
      constexpr float maxDelta = std::numeric_limits<float>::max();
      float i_delta = maxDelta;
      std::pair<int, int> i_nearestHigher(-1, -1);
      std::pair<float, int> nearest_distances(maxDelta, std::numeric_limits<int>::max());
      for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
        if (!nearestHigherOnSameLayer && (layerId == currentLayer))
          continue;
        const auto &tileOnLayer = d_hist[currentLayer];
        int etaWindow = 1;
        int phiWindow = 1;
        int etaBinMin = std::max(tileOnLayer.etaBin(clustersOnLayer.eta[i]) - etaWindow, 0);
        int etaBinMax = std::min(tileOnLayer.etaBin(clustersOnLayer.eta[i]) + etaWindow, nEtaBin);
        int phiBinMin = tileOnLayer.phiBin(clustersOnLayer.phi[i]) - phiWindow;
        int phiBinMax = tileOnLayer.phiBin(clustersOnLayer.phi[i]) + phiWindow;
        for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
          auto offset = ieta * nPhiBin;
          for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
            int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);
            if (algoVerbosity > 0) {
              std::cout << "Searching nearestHigher on " << currentLayer << " eta, phi: " << ieta << ", " << iphi_it
                        << " " << iphi << " " << offset << " " << (offset + iphi);
            }
            for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
              auto const &clustersOnOtherLayer = points[currentLayer];
              auto dist = maxDelta;
              auto dist_transverse = maxDelta;
              int dist_layers = std::abs(currentLayer - layerId);
              dist_transverse = distanceSqr(clustersOnLayer.r_over_absz[i] * clustersOnLayer.z[i],
                                            clustersOnOtherLayer.r_over_absz[otherClusterIdx] * clustersOnLayer.z[i],
                                            clustersOnLayer.phi[i],
                                            clustersOnOtherLayer.phi[otherClusterIdx]);
              // Add Z-scale to the final distance
              dist = dist_transverse;
              // TODO(rovere): in case of equal local density, the ordering in
              // the original CLUE3D implementaiton is bsaed on the index of
              // the LayerCclusters in the LayerClusterCollection. In this
              // case, the index is based on the ordering of the SOA indices.
              // bool foundHigher =
              //     (clustersOnOtherLayer.rho[otherClusterIdx] > clustersOnLayer.rho[i]) ||
              //     (clustersOnOtherLayer.rho[otherClusterIdx] == clustersOnLayer.rho[i] && otherClusterIdx > i);
              bool foundHigher =
                  (clustersOnOtherLayer.rho[otherClusterIdx] > clustersOnLayer.rho[i]) ||
                  (clustersOnOtherLayer.rho[otherClusterIdx] == clustersOnLayer.rho[i] &&
                   clustersOnOtherLayer.r_over_absz[otherClusterIdx] > clustersOnOtherLayer.r_over_absz[i]);
              if (algoVerbosity > 0) {
                std::cout << "Searching nearestHigher on " << currentLayer
                          << " with rho: " << clustersOnOtherLayer.rho[otherClusterIdx]
                          << " on layerIdxInSOA: " << currentLayer << ", " << otherClusterIdx
                          << " with distance: " << sqrt(dist) << " foundHigher: " << foundHigher;
              }
              if (foundHigher && dist <= i_delta) {
                // update i_delta
                i_delta = dist;
                nearest_distances = std::make_pair(sqrt(dist_transverse), dist_layers);
                // update i_nearestHigher
                i_nearestHigher = std::make_pair(currentLayer, otherClusterIdx);
              }
            }  // End of loop on clusters
          }    // End of loop on phi bins
        }      // End of loop on eta bins
      }        // End of loop on layers

      bool foundNearestInFiducialVolume = (i_delta != maxDelta);
      if (algoVerbosity > 0) {
        std::cout << "i_delta: " << i_delta << " passed: " << foundNearestInFiducialVolume << " "
                  << i_nearestHigher.first << " " << i_nearestHigher.second << " distances: " << nearest_distances.first
                  << ", " << nearest_distances.second;
      }
      if (foundNearestInFiducialVolume) {
        clustersOnLayer.delta[i] = nearest_distances;
        clustersOnLayer.nearestHigher[i] = i_nearestHigher;
      } else {
        // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        clustersOnLayer.delta[i] = std::make_pair(maxDelta, std::numeric_limits<int>::max());
        clustersOnLayer.nearestHigher[i] = {-1, -1};
      }
    }
  }
};

int KernelFindAndAssignClustersSoA(ClusterCollectionSerial &points,
                                   int algoVerbosity = 0,
                                   float criticalXYDistance = 1.8,  // cm
                                   float criticalZDistanceLyr = 5,
                                   float criticalDensity = 0.6,  // GeV
                                   float criticalSelfDensity = 0.15,
                                   float outlierMultiplier = 2.) {
  unsigned int nTracksters = 0;

  std::vector<std::pair<int, int>> localStack;
  auto critical_transverse_distance = criticalXYDistance;
  // find cluster seeds and outlier
  for (unsigned int clusterIdxSoA = 0; clusterIdxSoA < points.x.size(); ++clusterIdxSoA) {
    int layerId = points.layer[clusterIdxSoA];
    // initialize tracksterIndex
    points.tracksterIndex[clusterIdxSoA] = -1;
    bool isSeed = (points.delta[clusterIdxSoA].first > critical_transverse_distance ||
                   points.delta[clusterIdxSoA].second > criticalZDistanceLyr) &&
                  (points.rho[clusterIdxSoA] >= criticalDensity) &&
                  (points.energy[clusterIdxSoA] / points.rho[clusterIdxSoA] > criticalSelfDensity);
    if (!points.isSilicon[clusterIdxSoA]) {
      isSeed = (points.delta[clusterIdxSoA].first > points.radius[clusterIdxSoA] ||
                points.delta[clusterIdxSoA].second > criticalZDistanceLyr) &&
               (points.rho[clusterIdxSoA] >= criticalDensity) &&
               (points.energy[clusterIdxSoA] / points.rho[clusterIdxSoA] > criticalSelfDensity);
    }
    bool isOutlier = (points.delta[clusterIdxSoA].first > outlierMultiplier * critical_transverse_distance) &&
                     (points.rho[clusterIdxSoA] < criticalDensity);
    if (isSeed) {
      if (algoVerbosity > 0) {
        std::cout << "Found seed on Layer " << layerId << " SOAidx: " << clusterIdxSoA
                  << " assigned ClusterIdx: " << nTracksters;
      }
      points.tracksterIndex[clusterIdxSoA] = nTracksters++;
      points.isSeed[clusterIdxSoA] = true;
      localStack.emplace_back(layerId, clusterIdxSoA);
    } else if (!isOutlier) {
      auto [lyrIdx, soaIdx] = points.nearestHigher[clusterIdxSoA];
      if (algoVerbosity > 0) {
        std::cout << "Found follower on Layer " << layerId << " SOAidx: " << clusterIdxSoA
                  << " attached to cluster on layer: " << lyrIdx << " SOAidx: " << soaIdx;
      }
      if (lyrIdx >= 0)
        points.followers[soaIdx].emplace_back(layerId, clusterIdxSoA);
    } else {
      if (algoVerbosity > 0) {
        std::cout << "Found Outlier on Layer " << layerId << " SOAidx: " << clusterIdxSoA
                  << " with rho: " << points.rho[clusterIdxSoA] << " and delta: " << points.delta[clusterIdxSoA].first
                  << ", " << points.delta[clusterIdxSoA].second;
      }
    }
  }

  std::cout << "Number of Tracksters: " << nTracksters << std::endl;
  // Propagate cluster index
  while (!localStack.empty()) {
    auto [lyrIdx, soaIdx] = localStack.back();
    auto &thisSeed = points.followers[soaIdx];
    localStack.pop_back();

    // loop over followers
    for (auto [follower_lyrIdx, follower_soaIdx] : thisSeed) {
      // pass id to a follower
      points.tracksterIndex[follower_soaIdx] = points.tracksterIndex[soaIdx];
      // push this follower to localStack
      localStack.emplace_back(follower_lyrIdx, follower_soaIdx);
    }
  }

  return nTracksters;
};

int KernelFindAndAssignClusters(ClusterCollectionSerialOnLayers &points,
                                int algoVerbosity = 0,
                                float criticalXYDistance = 1.8,  // cm
                                float criticalZDistanceLyr = 5,
                                float criticalDensity = 0.6,  // GeV
                                float criticalSelfDensity = 0.15,
                                float outlierMultiplier = 2.) {
  constexpr int nLayers = TICLLayerTiles::constants_type_t::nLayers;
  unsigned int nTracksters = 0;

  std::vector<std::pair<int, int>> localStack;
  auto critical_transverse_distance = criticalXYDistance;
  // find cluster seeds and outlier
  for (unsigned int layer = 0; layer < nLayers; layer++) {
    auto &clustersOnLayer = points[layer];
    unsigned int numberOfClusters = clustersOnLayer.x.size();
    for (unsigned int i = 0; i < numberOfClusters; i++) {
      // initialize tracksterIndex
      clustersOnLayer.tracksterIndex[i] = -1;
      bool isSeed = (clustersOnLayer.delta[i].first > critical_transverse_distance ||
                     clustersOnLayer.delta[i].second > criticalZDistanceLyr) &&
                    (clustersOnLayer.rho[i] >= criticalDensity) &&
                    (clustersOnLayer.energy[i] / clustersOnLayer.rho[i] > criticalSelfDensity);
      if (!clustersOnLayer.isSilicon[i]) {
        isSeed = (clustersOnLayer.delta[i].first > clustersOnLayer.radius[i] ||
                  clustersOnLayer.delta[i].second > criticalZDistanceLyr) &&
                 (clustersOnLayer.rho[i] >= criticalDensity) &&
                 (clustersOnLayer.energy[i] / clustersOnLayer.rho[i] > criticalSelfDensity);
      }
      bool isOutlier = (clustersOnLayer.delta[i].first > outlierMultiplier * critical_transverse_distance) &&
                       (clustersOnLayer.rho[i] < criticalDensity);
      if (isSeed) {
        if (algoVerbosity > 0) {
          std::cout << "Found seed on Layer " << layer << " SOAidx: " << i << " assigned ClusterIdx: " << nTracksters;
        }
        clustersOnLayer.tracksterIndex[i] = nTracksters++;
        clustersOnLayer.isSeed[i] = true;
        localStack.emplace_back(layer, i);
      } else if (!isOutlier) {
        auto [lyrIdx, soaIdx] = clustersOnLayer.nearestHigher[i];
        if (algoVerbosity > 0) {
          std::cout << "Found follower on Layer " << layer << " SOAidx: " << i
                    << " attached to cluster on layer: " << lyrIdx << " SOAidx: " << soaIdx;
        }
        if (lyrIdx >= 0)
          points[lyrIdx].followers[soaIdx].emplace_back(layer, i);
      } else {
        if (algoVerbosity > 0) {
          std::cout << "Found Outlier on Layer " << layer << " SOAidx: " << i << " with rho: " << clustersOnLayer.rho[i]
                    << " and delta: " << clustersOnLayer.delta[i].first << ", " << clustersOnLayer.delta[i].second;
        }
      }
    }
  }

  std::cout << "Number of Tracksters: " << nTracksters << std::endl;
  // Propagate cluster index
  while (!localStack.empty()) {
    auto [lyrIdx, soaIdx] = localStack.back();
    auto &thisSeed = points[lyrIdx].followers[soaIdx];
    localStack.pop_back();

    // loop over followers
    for (auto [follower_lyrIdx, follower_soaIdx] : thisSeed) {
      // pass id to a follower
      points[follower_lyrIdx].tracksterIndex[follower_soaIdx] = points[lyrIdx].tracksterIndex[soaIdx];
      // push this follower to localStack
      localStack.emplace_back(follower_lyrIdx, follower_soaIdx);
    }
  }

  return nTracksters;
};

#endif
