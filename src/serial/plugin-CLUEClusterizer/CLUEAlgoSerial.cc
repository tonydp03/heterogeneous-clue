#include <iostream>
#include "DataFormats/PointsCloud.h"

#include "CLUEAlgoSerial.h"
#include "CLUEAlgoKernels.h"

void CLUEAlgoSerial::setup(PointsCloud const &host_pc) {
  // copy input variables
  d_points.x = host_pc.x;
  d_points.y = host_pc.y;
  d_points.layer = host_pc.layer;
  d_points.weight = host_pc.weight;
  d_points.n = host_pc.n;

  // resize output variables
  d_points.outResize(d_points.n);
}

void CLUEAlgoSerial::makeClusters(PointsCloud const &host_pc) {
  setup(host_pc);

  // calculate rho, delta and find seeds
  kernel_compute_histogram(hist_, d_points);
  kernel_calculate_density(hist_, d_points, dc_);
  kernel_calculate_distanceToHigher(hist_, d_points, outlierDeltaFactor_, dc_);
  kernel_findAndAssign_clusters(d_points, outlierDeltaFactor_, dc_, rhoc_);
}