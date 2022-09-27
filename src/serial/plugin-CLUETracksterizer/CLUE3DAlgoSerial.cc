#include <iostream>
#include "DataFormats/ClusterCollection.h"

#include "CLUE3DAlgoSerial.h"
#include "CLUE3DAlgoKernels.h"

void CLUE3DAlgoSerial::setup(ClusterCollection const &pc) {
  // this function should fill the ClusterCollection starting from PointsCloudSerial which has all the cluster indices
  // it should calculate the position of each cluster and the energy, and (??) store the number of hits.
  // Maybe we'll need also vector of rechit indices for each cluster

  // copy input variables
  d_clusters.x = pc.x;
  d_clusters.y = pc.y;
  d_clusters.layer = pc.layer;
  d_clusters.energy = pc.energy;
  d_clusters.n = pc.n;

  // resize output variables
  d_clusters.outResize(d_clusters.n);
}

void CLUE3DAlgoSerial::makeTracksters(ClusterCollection const &pc) {
  setup(pc);

  // calculate rho, delta and find seeds
  KernelComputeHistogram(hist_, d_clusters);
  KernelCalculateDensity(hist_, d_clusters, dc_);
  KernelComputeDistanceToHigher(hist_, d_clusters, outlierDeltaFactor_, dc_);
  KernelFindAndAssignClusters(d_clusters, outlierDeltaFactor_, dc_, rhoc_);
}