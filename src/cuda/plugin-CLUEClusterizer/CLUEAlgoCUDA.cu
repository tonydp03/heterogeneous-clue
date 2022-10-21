#include <math.h>
#include <limits>
#include <iostream>

// GPU Add
#include <cuda_runtime.h>
#include <cuda.h>
// for timing
#include <chrono>
#include <ctime>
// user include

#include "CLUEAlgoCUDA.h"
#include "CLUEAlgoKernels.h"
#include "CUDACore/cudaCheck.h"

void CLUEAlgoCUDA::init_device(int nPoints) {
  d_hist = cms::cuda::make_device_unique<LayerTilesCUDA[]>(NLAYERS, stream_);
  d_seeds = cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNSeeds>>(stream_);
  d_followers = cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNFollowers>[]>(nPoints, stream_);

  hist_ = d_hist.get();
  seeds_ = d_seeds.get();
  followers_ = d_followers.get();
}

void CLUEAlgoCUDA::setup(PointsCloud const& host_pc) {
  // copy input variables
  cudaCheck(cudaMemcpyAsync(
      d_points.x.get(), host_pc.x.data(), sizeof(float) * host_pc.x.size(), cudaMemcpyHostToDevice, stream_));
  cudaCheck(cudaMemcpyAsync(
      d_points.y.get(), host_pc.y.data(), sizeof(float) * host_pc.x.size(), cudaMemcpyHostToDevice, stream_));
  cudaCheck(cudaMemcpyAsync(
      d_points.layer.get(), host_pc.layer.data(), sizeof(int) * host_pc.x.size(), cudaMemcpyHostToDevice, stream_));
  cudaCheck(cudaMemcpyAsync(
      d_points.weight.get(), host_pc.weight.data(), sizeof(float) * host_pc.x.size(), cudaMemcpyHostToDevice, stream_));
  // initialize result and internal variables
  // // result variables
  //   cudaCheck(cudaMemsetAsync(d_points.rho.get(), 0x00, sizeof(float) * host_pc.x.size(), stream_));
  //   cudaCheck(cudaMemsetAsync(d_points.delta.get(), 0x00, sizeof(float) * host_pc.x.size(), stream_));
  //   cudaCheck(cudaMemsetAsync(d_points.nearestHigher.get(), 0x00, sizeof(int) * host_pc.x.size(), stream_));
  //   cudaCheck(cudaMemsetAsync(d_points.clusterIndex.get(), 0x00, sizeof(int) * host_pc.x.size(), stream_));
  //   cudaCheck(cudaMemsetAsync(d_points.isSeed.get(), 0x00, sizeof(int) * host_pc.x.size(), stream_));
  // algorithm internal variables
  //   cudaCheck(cudaMemsetAsync(d_hist.get(), 0x00, sizeof(LayerTilesCUDA) * NLAYERS, stream_));
  cudaCheck(cudaMemsetAsync(d_seeds.get(), 0x00, sizeof(cms::cuda::VecArray<int, maxNSeeds>), stream_));
  //   cudaCheck(cudaMemsetAsync(
  //       d_followers.get(), 0x00, sizeof(cms::cuda::VecArray<int, maxNFollowers>) * host_pc.x.size(), stream_));

  const dim3 blockSize(1024, 1, 1);
  dim3 gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize.x)), 1, 1);
  kernel_reset_followers<<<gridSize, blockSize, 0, stream_>>>(followers_, host_pc.x.size());
  gridSize.x = std::ceil(LayerTilesConstants::nRows * LayerTilesConstants::nColumns / static_cast<float>(blockSize.x));
  kernel_reset_hist<<<gridSize, blockSize, 0, stream_>>>(hist_);
}

void CLUEAlgoCUDA::makeClusters(PointsCloud const& host_pc) {
  setup(host_pc);
  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
  const dim3 blockSize(1024, 1, 1);
  const dim3 gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize.x)), 1, 1);
  kernel_compute_histogram<<<gridSize, blockSize, 0, stream_>>>(hist_, d_points.view(), host_pc.x.size());
  kernel_calculate_density<<<gridSize, blockSize, 0, stream_>>>(hist_, d_points.view(), dc_, host_pc.x.size());
  kernel_calculate_distanceToHigher<<<gridSize, blockSize, 0, stream_>>>(
      hist_, d_points.view(), outlierDeltaFactor_, dc_, host_pc.x.size());
  kernel_find_clusters<<<gridSize, blockSize, 0, stream_>>>(
      seeds_, followers_, d_points.view(), outlierDeltaFactor_, dc_, rhoc_, host_pc.x.size());

  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  const dim3 gridSize_nseeds(ceil(maxNSeeds / static_cast<float>(blockSize.x)), 1, 1);
  kernel_assign_clusters<<<gridSize_nseeds, blockSize, 0, stream_>>>(seeds_, followers_, d_points.view());
  cudaStreamSynchronize(stream_);
}
