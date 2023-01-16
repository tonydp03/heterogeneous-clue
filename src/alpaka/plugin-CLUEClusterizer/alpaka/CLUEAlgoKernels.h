#ifndef CLUEAlgo_Alpaka_Kernels_h
#define CLUEAlgo_Alpaka_Kernels_h

#include "AlpakaDataFormats/LayerTilesAlpaka.h"
#include "AlpakaDataFormats/alpaka/PointsCloudAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using pointsView = PointsCloudAlpaka::PointsCloudAlpakaView;

  struct KernelResetHist {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, LayerTilesAlpaka *d_hist) const {
      cms::alpakatools::for_each_element_in_grid(
          acc, LayerTilesConstants::nRows * LayerTilesConstants::nColumns, [&](uint32_t i) {
            for (int layerId = 0; layerId < NLAYERS; ++layerId)
              d_hist[layerId].clear(i);
          });
    }
  };

  struct KernelResetFollowers {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  cms::alpakatools::VecArray<int, maxNFollowers> *d_followers,
                                  uint32_t const &numberOfPoints) const {
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) { d_followers[i].reset(); });
    }
  };

  struct KernelComputeHistogram {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  LayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  uint32_t const &numberOfPoints) const {
      // push index of points into tiles
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) {
        d_hist[d_points->layer[i]].fill(acc, d_points->x[i], d_points->y[i], i);
      });
    }
  };

  struct KernelCalculateDensity {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  LayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  float dc,
                                  uint32_t const &numberOfPoints) const {
      const float dcSquared = dc * dc;
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) {
        float rhoi{0.f};
        int layeri = d_points->layer[i];
        float xi = d_points->x[i];
        float yi = d_points->y[i];
        //get search box
        int4 search_box = d_hist[layeri].searchBox(xi - dc, xi + dc, yi - dc, yi + dc);
        //loop over bins in the search box
        for (int xBin = search_box.x; xBin < search_box.y + 1; xBin++) {
          for (int yBin = search_box.z; yBin < search_box.w + 1; yBin++) {
            //get the id of this bin
            int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
            //get the size of this bin
            const auto &my_hist = d_hist[layeri][binId];
            int binSize = my_hist.size();
            //iterate inside this bin
            for (int binIter = 0; binIter < binSize; binIter++) {
              uint32_t j = my_hist[binIter];
              float xj = d_points->x[j];
              float yj = d_points->y[j];
              float dist_ij_squared = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
              rhoi += (int)(dist_ij_squared <= dcSquared) * (float((uint(i == j))*0.5f) +0.5f) * d_points->weight[j];
            }  // end of iterate inside this bin
          }
        }  // end of loop over bins in search box
        d_points->rho[i] = rhoi;
      });
    }
  };

  struct KernelComputeDistanceToHigher {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  LayerTilesAlpaka *d_hist,
                                  pointsView *d_points,
                                  float outlierDeltaFactor,
                                  float dc,
                                  uint32_t const &numberOfPoints) const {
      float dm = outlierDeltaFactor * dc;
      float dm_squared = dm * dm;
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) {
        int layeri = d_points->layer[i];
        float deltai = std::numeric_limits<float>::max();
        int nearestHigheri = -1;
        float xi = d_points->x[i];
        float yi = d_points->y[i];
        float rhoi = d_points->rho[i];
        // get search box
        int4 search_box = d_hist[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);
        // loop over all bins in the search box
        for (int xBin = search_box.x; xBin < search_box.y + 1; xBin++) {
          for (int yBin = search_box.z; yBin < search_box.w + 1; yBin++) {
            // get the id of this bin
            int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
            // get the size of this bin
            const auto &my_hist = d_hist[layeri][binId];
            int binSize = my_hist.size();
            //iterate inside this bin
            for (int binIter = 0; binIter < binSize; binIter++) {
              uint32_t j = my_hist[binIter];
              // query N'_{dm}(i)
              float xj = d_points->x[j];
              float yj = d_points->y[j];
              float dist_ij_squared = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
              bool foundHigher = (d_points->rho[j] > rhoi);
              // in the rare case where rho is the same, use detid
              foundHigher = foundHigher || ((d_points->rho[j] == rhoi) && (j > i));
              if (foundHigher && dist_ij_squared <= dm_squared) {
                // definition of N'_{dm}(i)
                // find the nearest point within N'_{dm}(i)
                if (dist_ij_squared < deltai) {
                  // update deltai and nearestHigheri
                  deltai = dist_ij_squared;
                  nearestHigheri = j;
                }
              }
            }  // end of iterate inside this bin
          }
        }  // end of loop over bins in search box
        d_points->delta[i] = std::sqrt(deltai);
        d_points->nearestHigher[i] = nearestHigheri;
      });
    }
  };

  struct KernelFindClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  cms::alpakatools::VecArray<int, maxNSeeds> *d_seeds,
                                  cms::alpakatools::VecArray<int, maxNFollowers> *d_followers,
                                  pointsView *d_points,
                                  float outlierDeltaFactor,
                                  float dc,
                                  float rhoc,
                                  uint32_t const &numberOfPoints) const {
      cms::alpakatools::for_each_element_in_grid(acc, numberOfPoints, [&](uint32_t i) {
        // initialize clusterIndex
        d_points->clusterIndex[i] = -1;
        // determine seed or outlier
        float deltai = d_points->delta[i];
        float rhoi = d_points->rho[i];
        bool isSeed = (deltai > dc) && (rhoi >= rhoc);
        bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

        if (isSeed) {
          // set isSeed as 1
          d_points->isSeed[i] = 1;
          d_seeds[0].push_back(acc, i);  // head of d_seeds
        } else {
          if (!isOutlier) {
            assert(d_points->nearestHigher[i] < static_cast<int>(numberOfPoints));
            // register as follower of its nearest higher
            d_followers[d_points->nearestHigher[i]].push_back(acc, i);
          }
          d_points->isSeed[i] = 0;
        }
      });
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  cms::alpakatools::VecArray<int, maxNSeeds> *d_seeds,
                                  cms::alpakatools::VecArray<int, maxNFollowers> *d_followers,
                                  pointsView *d_points) const {
      const auto &seeds = d_seeds[0];
      const auto nSeeds = seeds.size();
      cms::alpakatools::for_each_element_in_grid(acc, nSeeds, [&](uint32_t idxCls) {
        int localStack[localStackSizePerSeed] = {-1};
        int localStackSize = 0;

        // assign cluster to seed[idxCls]
        int idxThisSeed = seeds[idxCls];
        d_points->clusterIndex[idxThisSeed] = idxCls;
        // push_back idThisSeed to localStack
        // assert((localStackSize < localStackSizePerSeed));
        localStack[localStackSize] = idxThisSeed;
        localStackSize++;
        // process all elements in localStack
        while (localStackSize > 0) {
          // get last element of localStack
          // assert((localStackSize - 1 < localStackSizePerSeed));
          int idxEndOfLocalStack = localStack[localStackSize - 1];
          int temp_clusterIndex = d_points->clusterIndex[idxEndOfLocalStack];
          // pop_back last element of localStack
          // assert((localStackSize - 1 < localStackSizePerSeed));
          localStack[localStackSize - 1] = -1;
          localStackSize--;
          const auto &followers = d_followers[idxEndOfLocalStack];
          const auto followers_size = d_followers[idxEndOfLocalStack].size();
          // loop over followers of last element of localStack
          for (int j = 0; j < followers_size; ++j) {
            // pass id to follower
            int follower = followers[j];
            d_points->clusterIndex[follower] = temp_clusterIndex;
            // push_back follower to localStack
            // assert((localStackSize < localStackSizePerSeed));
            localStack[localStackSize] = follower;
            localStackSize++;
          }
        }
      });
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif