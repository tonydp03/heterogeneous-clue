#include <iostream>
#include "DataFormats/ClusterCollection.h"

#include "CLUE3DAlgoSerial.h"
#include "CLUE3DAlgoKernels.h"

void CLUE3DAlgoSerial::setupSoA(ClusterCollection const& pc, ClusterCollectionSerial& d_clustersSoA) {
  d_clustersSoA.x = pc.x;
  d_clustersSoA.y = pc.y;
  d_clustersSoA.z = pc.z;
  d_clustersSoA.eta = pc.eta;
  d_clustersSoA.phi = pc.phi;
  d_clustersSoA.r_over_absz = pc.r_over_absz;
  d_clustersSoA.radius = pc.radius;
  d_clustersSoA.layer = pc.layer;
  d_clustersSoA.energy = pc.energy;
  d_clustersSoA.isSilicon = pc.isSilicon;
  d_clustersSoA.outResize();
}
void CLUE3DAlgoSerial::setup(ClusterCollection const& pc, ClusterCollectionSerialOnLayers& d_clusters) {
  // copy input variables
  d_clusters.resize(ticl::TileConstants::nLayers);
  for (unsigned int i = 0; i < pc.x.size(); ++i) {
    d_clusters[pc.layer[i]].x.push_back(pc.x[i]);
    d_clusters[pc.layer[i]].y.push_back(pc.y[i]);
    d_clusters[pc.layer[i]].z.push_back(pc.z[i]);
    d_clusters[pc.layer[i]].eta.push_back(pc.eta[i]);
    d_clusters[pc.layer[i]].phi.push_back(pc.phi[i]);
    d_clusters[pc.layer[i]].r_over_absz.push_back(pc.r_over_absz[i]);
    d_clusters[pc.layer[i]].radius.push_back(pc.radius[i]);
    d_clusters[pc.layer[i]].layer.push_back(pc.layer[i]);
    d_clusters[pc.layer[i]].energy.push_back(pc.energy[i]);
    d_clusters[pc.layer[i]].isSilicon.push_back(pc.isSilicon[i]);
  }

  // resize output variables
  for (unsigned int layer = 0; layer < d_clusters.size(); ++layer) {
    d_clusters[layer].outResize();
  }
}

void CLUE3DAlgoSerial::makeTrackstersSoA(ClusterCollection const& pc, ClusterCollectionSerial& d_clustersSoA) {
  setupSoA(pc, d_clustersSoA);

  KernelComputeHistogramSoA(*histSoA_, d_clustersSoA);
  KernelCalculateDensitySoA(*histSoA_, d_clustersSoA);
  KernelComputeDistanceToHigherSoA(*histSoA_, d_clustersSoA);
  KernelFindAndAssignClustersSoA(d_clustersSoA);
  histSoA_->clear();
}

void CLUE3DAlgoSerial::makeTracksters(ClusterCollection const& pc, ClusterCollectionSerialOnLayers& d_clusters) {
  setup(pc, d_clusters);

  KernelComputeHistogram(*hist_, d_clusters);
  KernelCalculateDensity(*hist_, d_clusters);
  KernelComputeDistanceToHigher(*hist_, d_clusters);
  KernelFindAndAssignClusters(d_clusters);
  hist_->clear();
}
