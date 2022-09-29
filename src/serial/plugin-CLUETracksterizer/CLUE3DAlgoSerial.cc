#include <iostream>
#include "DataFormats/ClusterCollection.h"

#include "CLUE3DAlgoSerial.h"
#include "CLUE3DAlgoKernels.h"

void CLUE3DAlgoSerial::setup(ClusterCollection const &pc) {
  // this function should fill the ClusterCollection starting from PointsCloudSerial which has all the cluster indices
  // it should calculate the position of each cluster and the energy, and (??) store the number of hits.
  // Maybe we'll need also vector of rechit indices for each cluster

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

void CLUE3DAlgoSerial::makeTracksters(ClusterCollection const &pc) {
  setup(pc);

  // calculate rho, delta and find seeds
  KernelComputeHistogram(*hist_, d_clusters);
  KernelCalculateDensity(*hist_, d_clusters, dc_);
  KernelComputeDistanceToHigher(*hist_, d_clusters, outlierDeltaFactor_, dc_);
  KernelFindAndAssignClusters(d_clusters, outlierDeltaFactor_, dc_, rhoc_);
}
