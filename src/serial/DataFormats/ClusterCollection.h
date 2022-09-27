#ifndef Cluster_Collection_h
#define Cluster_Collection_h

#include <vector>

struct ClusterCollection {
  ClusterCollection() = default;

  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> r_over_absz;
  std::vector<float> radius;
  std::vector<int> layer;
  std::vector<float> energy;
  std::vector<int> isSilicon;

};

struct ClusterCollectionSerial {
  ClusterCollectionSerial() = default;

  void outResize() {
    auto nClusters = x.size();
    rho.resize(nClusters);
    delta.resize(nClusters);
    nearestHigher.resize(nClusters);
    tracksterIndex.resize(nClusters);
    followers.resize(nClusters);
    isSeed.resize(nClusters);
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
  std::vector<int> isSilicon;

  std::vector<float> rho;
  std::vector<std::pair<float, int>> delta;
  std::vector<std::pair<int, int>> nearestHigher;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  std::vector<int> tracksterIndex;

};

using ClusterCollectionSerialOnLayers = std::vector<ClusterCollectionSerial>;

#endif
