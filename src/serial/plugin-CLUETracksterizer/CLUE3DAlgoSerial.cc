#include <iostream>
#include "DataFormats/PointsCloud.h"
#include "DataFormats/ClusterCollection.h"

#include "CLUE3DAlgoSerial.h"
#include "CLUE3DAlgoKernels.h"

void CLUE3DAlgoSerial::setup(PointsCloudSerial const &pc) {
  // this function should fill the ClusterCollection starting from PointsCloudSerial which has all the cluster indices
  // it should calculate the position of each cluster and the energy, and (??) store the number of hits.
  // Maybe we'll need also vector of rechit indices for each cluster

  // // copy input variables
  // d_points.x = pc.x;
  // d_points.y = pc.y;
  // d_points.layer = pc.layer;
  // d_points.weight = pc.weight;
  // d_points.n = pc.n;

  // // resize output variables
  // d_points.outResize(d_points.n);
}

void CLUE3DAlgoSerial::makeTracksters(PointsCloudSerial const &pc) {
  setup(pc);

  // calculate rho, delta and find seeds
  KernelComputeHistogram(hist_, d_clusters);
  KernelCalculateDensity(hist_, d_clusters, dc_);
  KernelComputeDistanceToHigher(hist_, d_clusters, outlierDeltaFactor_, dc_);
  KernelFindAndAssignClusters(d_clusters, outlierDeltaFactor_, dc_, rhoc_);
}