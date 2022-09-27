#ifndef Cluster_Collection_h
#define Cluster_Collection_h

#include <vector>

struct ClusterCollection {
  ClusterCollection() = default;

  void outResize(unsigned int const& nClusters) {
    rho.resize(nClusters);
    delta.resize(nClusters);
    nearestHigher.resize(nClusters);
    tracksterIndex.resize(nClusters);
    followers.resize(nClusters);
    isSeed.resize(nClusters);
    n = nClusters;
  }

  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> r_over_absz;
  std::vector<float> radius;
  std::vector<int> layer;
  std::vector<float> energy;
  std::vector<int> nHits;  // don't know if it's necessary
  std::vector<int> isSilicon;

  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  std::vector<int> tracksterIndex;

  unsigned int n;
};

using ClusterCollectionOnLayers = std::vector<ClusterCollection>;

#endif
