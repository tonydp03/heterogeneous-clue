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

void CLUEAlgoSerial::makeClusters(PointsCloud const &host_pc,
                                  float const &dc,
                                  float const &rhoc,
                                  float const &outlierDeltaFactor) {
  setup(host_pc);

  // calculate rho, delta and find seeds
  
  KernelComputeHistogram(*hist_, d_points);
  KernelCalculateDensity(*hist_, d_points, dc);
  KernelComputeDistanceToHigher(*hist_, d_points, outlierDeltaFactor, dc);
  KernelFindAndAssignClusters(d_points, outlierDeltaFactor, dc, rhoc);

  for (unsigned int index = 0; index < hist_->size(); ++index) {
    (*hist_)[index].clear();
  }
}