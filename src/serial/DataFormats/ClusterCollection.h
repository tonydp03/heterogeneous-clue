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
  std::vector<int> layer;
  std::vector<float> energy;
  std::vector<int> nHits;  // don't know if it's necessary

  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  std::vector<int> tracksterIndex;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])

  unsigned int n;
};

#endif