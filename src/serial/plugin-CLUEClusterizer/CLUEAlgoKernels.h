#ifndef CLUEAlgo_Serial_Kernels_h
#define CLUEAlgo_Serial_Kernels_h

#include "DataFormats/LayerTilesSerial.h"
#include "DataFormats/PointsCloud.h"

inline float distance(PointsCloudSerial &points, int i, int j) {
  // 2-d distance on the layer
  if (points.layer[i] == points.layer[j]) {
    const float dx = points.x[i] - points.x[j];
    const float dy = points.y[i] - points.y[j];
    return std::sqrt(dx * dx + dy * dy);
  } else {
    return std::numeric_limits<float>::max();
  }
}

void kernel_compute_histogram(LayerTilesSerial *d_hist, PointsCloudSerial &points) {
  for (unsigned int i = 0; i < points.n; i++) {
    // push index of points into tiles
    d_hist[points.layer[i]].fill(points.x[i], points.y[i], i);
  }
};

void kernel_calculate_density(LayerTilesSerial *d_hist, PointsCloudSerial &points, float dc) {
  // loop over all points
  for (unsigned int i = 0; i < points.n; i++) {
    LayerTilesSerial &lt = d_hist[points.layer[i]];

    // get search box
    std::array<int, 4> search_box =
        lt.searchBox(points.x[i] - dc, points.x[i] + dc, points.y[i] - dc, points.y[i] + dc);

    // loop over bins in the search box
    for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
      for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = lt[binId].size();

        // iterate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          unsigned int j = lt[binId][binIter];
          // query N_{dc}(i)
          float dist_ij = distance(points, i, j);
          if (dist_ij <= dc) {
            // sum weights within N_{dc}(i)
            points.rho[i] += (i == j ? 1.f : 0.5f) * points.weight[j];
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
  }    // end of loop over points
};

void kernel_calculate_distanceToHigher(LayerTilesSerial *d_hist,
                                       PointsCloudSerial &points,
                                       float outlierDeltaFactor,
                                       float dc) {
  // loop over all points
  float dm = outlierDeltaFactor * dc;
  for (unsigned int i = 0; i < points.n; i++) {
    // default values of delta and nearest higher for i
    float delta_i = std::numeric_limits<float>::max();
    int nearestHigher_i = -1;
    float xi = points.x[i];
    float yi = points.y[i];
    float rho_i = points.rho[i];

    // get search box
    LayerTilesSerial &lt = d_hist[points.layer[i]];
    std::array<int, 4> search_box = lt.searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

    // loop over all bins in the search box
    for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
      for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = lt[binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          unsigned int j = lt[binId][binIter];
          // query N'_{dm}(i)
          bool foundHigher = (points.rho[j] > rho_i);
          // in the rare case where rho is the same, use detid
          // foundHigher = foundHigher || ((points.rho[j] == rho_i) && (j > i));
          float dist_ij = distance(points, i, j);
          if (foundHigher && dist_ij <= dm) {  // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij < delta_i) {
              // update delta_i and nearestHigher_i
              delta_i = dist_ij;
              nearestHigher_i = j;
            }
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box

    points.delta[i] = delta_i;
    points.nearestHigher[i] = nearestHigher_i;
  }  // end of loop over points
};

void kernel_findAndAssign_clusters(PointsCloudSerial &points,
                                   cms::cuda::VecArray<int, 100000> *d_seeds,
                                   cms::cuda::VecArray<int, 128> *d_followers,
                                   float outlierDeltaFactor,
                                   float dc,
                                   float rhoc) {
  int nClusters = 0;
  int localStackSizePerSeed = 128;
  // find cluster seeds and outlier
  // std::vector<int> localStack;
  // loop over all points
  for (unsigned int i = 0; i < points.n; i++) {
    // initialize clusterIndex
    points.clusterIndex[i] = -1;

    float deltai = points.delta[i];
    float rhoi = points.rho[i];

    // determine seed or outlier
    bool isSeed = (deltai > dc) and (rhoi >= rhoc);
    bool isOutlier = (deltai > outlierDeltaFactor * dc) and (rhoi < rhoc);
    if (isSeed) {
      // set isSeed as 1
      points.isSeed[i] = 1;
      // set cluster id
      points.clusterIndex[i] = nClusters;
      // increment number of clusters
      nClusters++;
      // add seed into local stack
      // localStack.push_back(i);
      d_seeds->push_back(i);
    } else if (!isOutlier) {
      // register as follower at its nearest higher
      // points.followers[points.nearestHigher[i]].push_back(i);
      d_followers[points.nearestHigher[i]].push_back(i);
    }
  }

  // expend clusters from seeds
  const auto &seeds = d_seeds[0];
  const auto nSeeds = seeds.size();
  int localStack[localStackSizePerSeed] = {-1};
  int localStackSize = 0;
  for (int i = 0; i < nSeeds; i++) {
    // assign cluster to seed[idxCls]
    int idxThisSeed = seeds[i];
    points.clusterIndex[idxThisSeed] = i;
    // push_back idThisSeed to localStack
    // assert((localStackSize < localStackSizePerSeed));

    localStack[localStackSize] = idxThisSeed;
    localStackSize++;
    // process all elements in localStack
    while (localStackSize > 0) {
      // get last element of localStack
      // assert((localStackSize - 1 < localStackSizePerSeed));
      int idxEndOfLocalStack = localStack[localStackSize - 1];
      int temp_clusterIndex = points.clusterIndex[idxEndOfLocalStack];
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
        points.clusterIndex[follower] = temp_clusterIndex;
        // push_back follower to localStack
        // assert((localStackSize < localStackSizePerSeed));
        localStack[localStackSize] = follower;
        localStackSize++;
      }
    }
  }
  // while (!localStack.empty()) {
  //   int i = localStack.back();
  //   auto &followers = points.followers[i];
  //   localStack.pop_back();

  //   // loop over followers
  //   for (int j : followers) {
  //     // pass id from i to a i's follower
  //     points.clusterIndex[j] = points.clusterIndex[i];
  //     // push this follower to localStack
  //     localStack.push_back(j);
  //   }
  // }
};

#endif
