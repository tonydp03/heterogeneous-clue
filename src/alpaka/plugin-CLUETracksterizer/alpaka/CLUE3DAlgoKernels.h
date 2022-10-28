#ifndef CLUE3DAlgo_Alpaka_Kernels_h
#define CLUE3DAlgo_Alpaka_Kernels_h

#include "DataFormats/Math/deltaR.h"
#include "DataFormats/Math/deltaPhi.h"
#include "DataFormats/Common.h"

#include "AlpakaDataFormats/TICLLayerTilesAlpaka.h"
#include "AlpakaDataFormats/alpaka/ClusterCollectionAlpaka.h"

// #include <assert.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using pointsView = ClusterCollectionAlpaka::ClusterCollectionAlpakaView;

  struct KernelComputeHistogram {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  TICLLayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  uint32_t const &numberOfPoints) const {
      // push index of points into tiles
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t clusterIdx) {
        d_hist[d_points->layer[clusterIdx]].fill(
            acc, std::abs(d_points->eta[clusterIdx]), d_points->phi[clusterIdx], clusterIdx);
      });
    }
  };

  struct KernelCalculateDensity {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  TICLLayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  uint32_t const &numberOfPoints,
                                  int densitySiblingLayers = 3,
                                  int densityXYDistanceSqr = 3.24,
                                  float kernelDensityFactor = 0.2,
                                  bool densityOnSameLayer = false) const {
      int nEtaBin = ticl::TileConstants::nEtaBins;
      int nPhiBin = ticl::TileConstants::nPhiBins;
      int nLayers = ticl::TileConstants::nLayers;

      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t clusterIdx) {
        int layerId = d_points->layer[clusterIdx];

        auto isReachable = [](float r0, float r1, float phi0, float phi1, float delta_sqr) -> bool {
          // TODO(rovere): import reco::deltaPhi implementation as well
          auto delta_phi = reco::deltaPhi(phi0, phi1);
          return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi < delta_sqr;
        };

        // We need to partition the two sides of the HGCAL detector
        int lastLayerPerSide = nLayers / 2;
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
          const auto &tileOnLayer = d_hist[currentLayer];
          bool onSameLayer = (currentLayer == layerId);

          const int etaWindow = 2;
          const int phiWindow = 2;
          int etaBinMin = std::max(tileOnLayer.etaBin(d_points->eta[clusterIdx]) - etaWindow, 0);
          int etaBinMax = std::min(tileOnLayer.etaBin(d_points->eta[clusterIdx]) + etaWindow, nEtaBin);
          int phiBinMin = tileOnLayer.phiBin(d_points->phi[clusterIdx]) - phiWindow;
          int phiBinMax = tileOnLayer.phiBin(d_points->phi[clusterIdx]) + phiWindow;

          for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
            auto offset = ieta * nPhiBin;

            for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
              int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);

              for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
                bool reachable = false;
                // Still differentiate between silicon and Scintillator.
                // Silicon has yet to be studied further.
                if (d_points->isSilicon[clusterIdx]) {
                  reachable = isReachable(d_points->r_over_absz[clusterIdx] * d_points->z[clusterIdx],
                                          d_points->r_over_absz[otherClusterIdx] * d_points->z[clusterIdx],
                                          d_points->phi[clusterIdx],
                                          d_points->phi[otherClusterIdx],
                                          densityXYDistanceSqr);
                } else {
                  reachable = isReachable(d_points->r_over_absz[clusterIdx] * d_points->z[clusterIdx],
                                          d_points->r_over_absz[otherClusterIdx] * d_points->z[clusterIdx],
                                          d_points->phi[clusterIdx],
                                          d_points->phi[otherClusterIdx],
                                          d_points->radius[clusterIdx] * d_points->radius[clusterIdx]);
                }

                if (reachable) {
                  float factor_same_layer_different_cluster = (onSameLayer && !densityOnSameLayer) ? 0.f : 1.f;
                  auto energyToAdd =
                      ((clusterIdx == otherClusterIdx) ? 1.f
                                                       : kernelDensityFactor * factor_same_layer_different_cluster) *
                      d_points->energy[otherClusterIdx];
                  d_points->rho[clusterIdx] += energyToAdd;
                }
              }  // end of loop on possible compatible clusters
            }    // end of loop over phi-bin region
          }      // end of loop over eta-bin region
        }        // end of loop on the sibling layers
      });
    }
  };

  struct KernelComputeDistanceToHigher {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  TICLLayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  uint32_t const &numberOfPoints,
                                  int densitySiblingLayers = 3,
                                  bool nearestHigherOnSameLayer = false) const {
      int nEtaBin = ticl::TileConstants::nEtaBins;
      int nPhiBin = ticl::TileConstants::nPhiBins;
      int nLayers = ticl::TileConstants::nLayers;

      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t clusterIdx) {
        int layerId = d_points->layer[clusterIdx];

        auto distanceSqr = [](float r0, float r1, float phi0, float phi1) -> float {
          auto delta_phi = reco::deltaPhi(phi0, phi1);
          return (r0 - r1) * (r0 - r1) + r1 * r1 * delta_phi * delta_phi;
        };

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
        float maxDelta = std::numeric_limits<float>::max();
        float i_delta = maxDelta;
        int i_nearestHigher = -1;
        std::pair<float, int> nearest_distances(maxDelta, std::numeric_limits<int>::max());
        for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
          if (!nearestHigherOnSameLayer && (layerId == currentLayer))
            continue;
          const auto &tileOnLayer = d_hist[currentLayer];
          int etaWindow = 1;
          int phiWindow = 1;
          int etaBinMin = std::max(tileOnLayer.etaBin(d_points->eta[clusterIdx]) - etaWindow, 0);
          int etaBinMax = std::min(tileOnLayer.etaBin(d_points->eta[clusterIdx]) + etaWindow, nEtaBin);
          int phiBinMin = tileOnLayer.phiBin(d_points->phi[clusterIdx]) - phiWindow;
          int phiBinMax = tileOnLayer.phiBin(d_points->phi[clusterIdx]) + phiWindow;
          for (int ieta = etaBinMin; ieta <= etaBinMax; ++ieta) {
            auto offset = ieta * nPhiBin;
            for (int iphi_it = phiBinMin; iphi_it <= phiBinMax; ++iphi_it) {
              int iphi = ((iphi_it % nPhiBin + nPhiBin) % nPhiBin);

              for (auto otherClusterIdx : tileOnLayer[offset + iphi]) {
                //              auto const &clustersOnOtherLayer = points[currentLayer];
                auto dist = maxDelta;
                auto dist_transverse = maxDelta;
                int dist_layers = std::abs(currentLayer - layerId);
                dist_transverse = distanceSqr(d_points->r_over_absz[clusterIdx] * d_points->z[clusterIdx],
                                              d_points->r_over_absz[otherClusterIdx] * d_points->z[clusterIdx],
                                              d_points->phi[clusterIdx],
                                              d_points->phi[otherClusterIdx]);
                // Add Z-scale to the final distance
                dist = dist_transverse;
                // TODO(rovere): in case of equal local density, the ordering in
                // the original CLUE3D implementaiton is bsaed on the index of
                // the LayerCclusters in the LayerClusterCollection. In this
                // case, the index is based on the ordering of the SOA indices.
                bool foundHigher = (d_points->rho[otherClusterIdx] > d_points->rho[clusterIdx]);  //||
                // (d_points->rho[otherClusterIdx] == d_points->rho[clusterIdx] && otherClusterIdx > clusterIdx);

                if (foundHigher && dist <= i_delta) {
                  // update i_delta
                  i_delta = dist;
                  nearest_distances.first = sqrt(dist_transverse);
                  nearest_distances.second = dist_layers;
                  // update i_nearestHigher
                  i_nearestHigher = otherClusterIdx;
                }
              }  // End of loop on clusters
            }    // End of loop on phi bins
          }      // End of loop on eta bins
        }        // End of loop on layers

        bool foundNearestInFiducialVolume = (i_delta != maxDelta);

        if (foundNearestInFiducialVolume) {
          d_points->delta[clusterIdx].first = nearest_distances.first;
          d_points->delta[clusterIdx].second = nearest_distances.second;
          d_points->nearestHigher[clusterIdx] = i_nearestHigher;
        } else {
          // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
          // we can safely maximize delta to be maxDelta
          d_points->delta[clusterIdx].first = maxDelta;
          d_points->delta[clusterIdx].second = std::numeric_limits<int>::max();
          d_points->nearestHigher[clusterIdx] = -1;
        }
      });
    }
  };

  struct KernelFindClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  cms::alpakatools::VecArray<int, ticl::maxNSeeds> *d_seeds,
                                  cms::alpakatools::VecArray<int, ticl::maxNFollowers> *d_followers,
                                  pointsView *d_points,
                                  uint32_t const &numberOfPoints,
                                  float criticalXYDistance = 1.8,  // cm
                                  float criticalZDistanceLyr = 5,
                                  float criticalDensity = 0.6,  // GeV
                                  float criticalSelfDensity = 0.15,
                                  float outlierMultiplier = 2.) const {
      auto critical_transverse_distance = criticalXYDistance;
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t clusterIdx) {
        // initialize tracksterIndex
        d_points->tracksterIndex[clusterIdx] = -1;
        bool isSeed = (d_points->delta[clusterIdx].first > critical_transverse_distance ||
                       d_points->delta[clusterIdx].second > criticalZDistanceLyr) &&
                      (d_points->rho[clusterIdx] >= criticalDensity) &&
                      (d_points->energy[clusterIdx] / d_points->rho[clusterIdx] > criticalSelfDensity);
        if (!d_points->isSilicon[clusterIdx]) {
          isSeed = (d_points->delta[clusterIdx].first > d_points->radius[clusterIdx] ||
                    d_points->delta[clusterIdx].second > criticalZDistanceLyr) &&
                   (d_points->rho[clusterIdx] >= criticalDensity) &&
                   (d_points->energy[clusterIdx] / d_points->rho[clusterIdx] > criticalSelfDensity);
        }
        bool isOutlier = (d_points->delta[clusterIdx].first > outlierMultiplier * critical_transverse_distance) &&
                         (d_points->rho[clusterIdx] < criticalDensity);
        if (isSeed) {
          d_seeds[0].push_back(acc, clusterIdx);
          d_points->isSeed[clusterIdx] = 1;
        } else {
          if (!isOutlier) {
            auto soaIdx = d_points->nearestHigher[clusterIdx];
            if (soaIdx >= 0)
              d_followers[soaIdx].push_back(acc, clusterIdx);
          }
        }
      });
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  cms::alpakatools::VecArray<int, ticl::maxNSeeds> *d_seeds,
                                  cms::alpakatools::VecArray<int, ticl::maxNFollowers> *d_followers,
                                  pointsView *d_points) const {
      const auto &seeds = d_seeds[0];
      const auto nSeeds = seeds.size();
      cms::alpakatools::for_each_element_in_grid(acc, nSeeds, [&](uint32_t idxCls) {
        int localStack[ticl::localStackSizePerSeed] = {-1};
        int localStackSize = 0;

        // assign cluster to seed[idxCls]
        int idxThisSeed = seeds[idxCls];
        d_points->tracksterIndex[idxThisSeed] = idxCls;
        // push_back idThisSeed to localStack
        assert((localStackSize < ticl::localStackSizePerSeed));

        localStack[localStackSize] = idxThisSeed;
        localStackSize++;
        // process all elements in localStack
        while (localStackSize > 0) {
          // get last element of localStack
          assert((localStackSize - 1 < ticl::localStackSizePerSeed));
          int idxEndOfLocalStack = localStack[localStackSize - 1];
          int temp_tracksterIndex = d_points->tracksterIndex[idxEndOfLocalStack];
          // pop_back last element of localStack
          assert((localStackSize - 1 < ticl::localStackSizePerSeed));
          localStack[localStackSize - 1] = -1;
          localStackSize--;

          // loop over followers of last element of localStack
          for (int j : d_followers[idxEndOfLocalStack]) {
            // pass id to follower
            d_points->tracksterIndex[j] = temp_tracksterIndex;
            // push_back follower to localStack
            assert((localStackSize < ticl::localStackSizePerSeed));
            localStack[localStackSize] = j;
            localStackSize++;
          }
        }
      });
    }
  };

  struct KernelPrintNTracksters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, pointsView *d_points, uint32_t const &numberOfPoints) const {
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) {
        auto numberOfTracksters = d_points->tracksterIndex[i];
        for (uint32_t j = 0; j < numberOfPoints; j++) {
          if (d_points->tracksterIndex[j] > numberOfTracksters)
            numberOfTracksters = d_points->tracksterIndex[j];
        }
        numberOfTracksters++;

        printf("Number of Tracksters: %d", numberOfTracksters);
        printf("\n");
      });
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
