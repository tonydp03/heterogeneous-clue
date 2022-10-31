#include <iostream>
#include "DataFormats/PointsCloud.h"

#include "CLUEAlgoSerial.h"
#include "CLUEAlgoKernels.h"

void CLUEAlgoSerial::setup(PointsCloud const &host_pc, PointsCloudSerial &d_points) {
  // copy input variables
  d_points.x = host_pc.x;
  d_points.y = host_pc.y;
  d_points.layer = host_pc.layer;
  d_points.weight = host_pc.weight;

  // resize output variables
  d_points.outResize();
}

void CLUEAlgoSerial::makeClusters(PointsCloud const &host_pc,
                                  PointsCloudSerial &d_points,
                                  float const &dc,
                                  float const &rhoc,
                                  float const &outlierDeltaFactor) {
  setup(host_pc, d_points);

  // calculate rho, delta and find seeds

  kernel_compute_histogram(*hist_, d_points);
  kernel_calculate_density(*hist_, d_points, dc);
  kernel_calculate_distanceToHigher(*hist_, d_points, outlierDeltaFactor, dc);
  kernel_findAndAssign_clusters(d_points, outlierDeltaFactor, dc, rhoc);

  for (unsigned int index = 0; index < hist_->size(); ++index) {
    (*hist_)[index].clear();
  }
}