#ifndef CLUE3DAlgo_SYCL_Kernels_h
#define CLUE3DAlgo_SYCL_Kernels_h

#include "DataFormats/Math/deltaR.h"
#include "DataFormats/Math/deltaPhi.h"
#include "DataFormats/Common.h"

#include "SYCLDataFormats/TICLLayerTilesSYCL.h"
#include "SYCLDataFormats/ClusterCollectionSYCL.h"

using pointsView = ClusterCollectionSYCL::ClusterCollectionSYCLView;

void kernel_reset_followers(cms::sycltools::VecArray<int, ticl::maxNFollowers> *d_followers,
                            uint32_t const &numberOfPoints,
                            sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (i < numberOfPoints) {
    d_followers[i].reset();
  }
}

void kernel_reset_hist(TICLLayerTilesSYCL *d_hist, sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (i < ticl::TileConstants::nBins) {
    for (int layerId = 0; layerId < ticl::TileConstants::nLayers; ++layerId)
      d_hist[layerId].clear(i);
  }
}

void kernel_compute_histogram(TICLLayerTilesSYCL *d_hist,
                              pointsView *d_points,
                              uint32_t const &numberOfPoints,
                              sycl::nd_item<1> item) {
  auto clusterIdx = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (clusterIdx < numberOfPoints) {
    d_hist[d_points->layer[clusterIdx]].fill(
        sycl::abs(d_points->eta[clusterIdx]), d_points->phi[clusterIdx], clusterIdx);
  }
}

void kernel_calculate_density(TICLLayerTilesSYCL *d_hist,
                              pointsView *d_points,
                              uint32_t const &numberOfPoints,
                              sycl::nd_item<1> item,
                              int densitySiblingLayers = 3,
                              float densityXYDistanceSqr = 3.24,
                              float kernelDensityFactor = 0.2,
                              bool densityOnSameLayer = false) {
  constexpr int nEtaBin = ticl::TileConstants::nEtaBins;
  constexpr int nPhiBin = ticl::TileConstants::nPhiBins;
  constexpr int nLayers = ticl::TileConstants::nLayers;
  auto clusterIdx = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

  if (clusterIdx < numberOfPoints) {
    int layerId = d_points->layer[clusterIdx];
    float rhoi{0.f};

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
      minLayer = sycl::max(layerId - densitySiblingLayers, minLayer);
      maxLayer = sycl::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
    } else {
      minLayer = sycl::max(layerId - densitySiblingLayers, lastLayerPerSide);
      maxLayer = sycl::min(layerId + densitySiblingLayers, maxLayer);
    }

    for (int currentLayer = minLayer; currentLayer <= maxLayer; currentLayer++) {
      const auto &tileOnLayer = d_hist[currentLayer];
      bool onSameLayer = (currentLayer == layerId);

      const int etaWindow = 2;
      const int phiWindow = 2;
      int etaBinMin = sycl::max(tileOnLayer.etaBin(d_points->eta[clusterIdx]) - etaWindow, 0);
      int etaBinMax = sycl::min(tileOnLayer.etaBin(d_points->eta[clusterIdx]) + etaWindow, nEtaBin);
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
                  ((clusterIdx == otherClusterIdx) ? 1.f : kernelDensityFactor * factor_same_layer_different_cluster) *
                  d_points->energy[otherClusterIdx];
              rhoi += energyToAdd;
            }
          }  // end of loop on possible compatible clusters
        }    // end of loop over phi-bin region
      }      // end of loop over eta-bin region
    }        // end of loop on the sibling layers
    d_points->rho[clusterIdx] = rhoi;
  }
}

void kernel_compute_distance_to_higher(TICLLayerTilesSYCL *d_hist,
                                       pointsView *d_points,
                                       uint32_t numberOfPoints,
                                       sycl::nd_item<1> item,
                                       int densitySiblingLayers = 3,
                                       bool nearestHigherOnSameLayer = false) {
  constexpr int nEtaBin = ticl::TileConstants::nEtaBins;
  constexpr int nPhiBin = ticl::TileConstants::nPhiBins;
  constexpr int nLayers = ticl::TileConstants::nLayers;

  auto clusterIdx = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (clusterIdx < numberOfPoints) {
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
      minLayer = sycl::max(layerId - densitySiblingLayers, minLayer);
      maxLayer = sycl::min(layerId + densitySiblingLayers, lastLayerPerSide - 1);
    } else {
      minLayer = sycl::max(layerId - densitySiblingLayers, lastLayerPerSide + 1);
      maxLayer = sycl::min(layerId + densitySiblingLayers, maxLayer);
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
      int etaBinMin = sycl::max(tileOnLayer.etaBin(d_points->eta[clusterIdx]) - etaWindow, 0);
      int etaBinMax = sycl::min(tileOnLayer.etaBin(d_points->eta[clusterIdx]) + etaWindow, nEtaBin);
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
            int dist_layers = sycl::abs(currentLayer - layerId);
            dist_transverse = distanceSqr(d_points->r_over_absz[clusterIdx] * d_points->z[clusterIdx],
                                          d_points->r_over_absz[otherClusterIdx] * d_points->z[clusterIdx],
                                          d_points->phi[clusterIdx],
                                          d_points->phi[otherClusterIdx]);
            // Add Z-scale to the final distance
            dist = dist_transverse;
            bool foundHigher = (d_points->rho[otherClusterIdx] > d_points->rho[clusterIdx]);  //||
            // (d_points->rho[otherClusterIdx] == d_points->rho[clusterIdx] && otherClusterIdx > clusterIdx);

            if (foundHigher && dist <= i_delta) {
              // update i_delta
              i_delta = dist;
              nearest_distances.first = sycl::sqrt(dist_transverse);
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
  }
}

void kernel_find_tracksters(cms::sycltools::VecArray<int, ticl::maxNSeeds> *d_seeds,
                            cms::sycltools::VecArray<int, ticl::maxNFollowers> *d_followers,
                            pointsView *d_points,
                            uint32_t const &numberOfPoints,
                            sycl::nd_item<1> item,
                            float criticalXYDistance = 1.8,  // cm
                            float criticalZDistanceLyr = 5,
                            float criticalDensity = 0.6,  // GeV
                            float criticalSelfDensity = 0.15,
                            float outlierMultiplier = 2.) {
  auto critical_transverse_distance = criticalXYDistance;
  auto clusterIdx = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (clusterIdx < numberOfPoints) {
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
      d_seeds[0].push_back(clusterIdx);
      d_points->isSeed[clusterIdx] = 1;
    } else {
      if (!isOutlier) {
        auto soaIdx = d_points->nearestHigher[clusterIdx];
        if (soaIdx >= 0)
          d_followers[soaIdx].push_back(clusterIdx);
      }
      d_points->isSeed[clusterIdx] = 0;
    }
  }
}

void kernel_assign_tracksters(const cms::sycltools::VecArray<int, ticl::maxNSeeds> *d_seeds,
                              const cms::sycltools::VecArray<int, ticl::maxNFollowers> *d_followers,
                              pointsView *d_points,
                              sycl::nd_item<1> item) {
  const auto &seeds = d_seeds[0];
  const unsigned long nSeeds = seeds.size();
  auto idxCls = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (idxCls < nSeeds) {
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
      // assert((localStackSize - 1 < ticl::localStackSizePerSeed));
      int idxEndOfLocalStack = localStack[localStackSize - 1];
      int temp_tracksterIndex = d_points->tracksterIndex[idxEndOfLocalStack];
      // pop_back last element of localStack
      assert((localStackSize - 1 < ticl::localStackSizePerSeed));
      localStack[localStackSize - 1] = -1;
      localStackSize--;
      const auto &followers = d_followers[idxEndOfLocalStack];
      const auto followers_size = d_followers[idxEndOfLocalStack].size();
      // loop over followers of last element of localStack
      for (int j = 0; j < followers_size; ++j) {
        // pass id to follower
        int follower = followers[j];
        d_points->tracksterIndex[j] = temp_tracksterIndex;
        // push_back follower to localStack
        assert((localStackSize < ticl::localStackSizePerSeed));
        localStack[localStackSize] = follower;
        localStackSize++;
      }
    }
  }
}

void kernel_print_ntracksters(pointsView *d_points, uint32_t const &numberOfPoints, sycl::nd_item<1> item, sycl::stream out) {
  auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (i < numberOfPoints) {
    auto numberOfTracksters = d_points->tracksterIndex[i];
    for (uint32_t j = 0; j < numberOfPoints; j++) {
      if (d_points->tracksterIndex[j] > numberOfTracksters)
        numberOfTracksters = d_points->tracksterIndex[j];
    }
    numberOfTracksters++;

    out << "Number of Tracksters: " << numberOfTracksters << sycl::endl;
  }
}

#endif