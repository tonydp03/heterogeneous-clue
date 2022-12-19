#include <CL/sycl.hpp>

#include "DataFormats/PointsCloud.h"

#include "CLUEAlgoSYCL.h"
#include "CLUEAlgoKernels.h"

CLUEAlgoSYCL::CLUEAlgoSYCL(float const &dc,
                           float const &rhoc,
                           float const &outlierDeltaFactor,
                           sycl::queue const &stream)
    : dc_{dc},
      rhoc_{rhoc},
      outlierDeltaFactor_{outlierDeltaFactor},
      d_hist{cms::sycltools::make_device_unique_uninitialized<LayerTilesSYCL[]>(NLAYERS, stream)},
      d_seeds{cms::sycltools::make_device_unique_uninitialized<cms::sycltools::VecArray<int, maxNSeeds>>(stream)},
      d_followers{cms::sycltools::make_device_unique_uninitialized<cms::sycltools::VecArray<int, maxNFollowers>[]>(
          reserve, stream)} {}

void CLUEAlgoSYCL::setup(PointsCloud const &host_pc, PointsCloudSYCL &d_points, sycl::queue &stream) {
  // input variables
  stream.memcpy(d_points.x.get(), host_pc.x.data(), sizeof(float) * host_pc.x.size());
  stream.memcpy(d_points.y.get(), host_pc.y.data(), sizeof(float) * host_pc.x.size());
  stream.memcpy(d_points.layer.get(), host_pc.layer.data(), sizeof(int) * host_pc.x.size());
  stream.memcpy(d_points.weight.get(), host_pc.weight.data(), sizeof(float) * host_pc.x.size());
  // result and internal variables
  stream.memset(d_seeds.get(), 0x00, sizeof(cms::sycltools::VecArray<int, maxNSeeds>));
  const sycl::range<1> blockSize(1024);
  sycl::range<1> gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize[0])));
  auto workDiv = sycl::nd_range<1>(gridSize * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_followers_kernel = d_followers.get();
    auto num_points = host_pc.x.size();
    cgh.parallel_for(workDiv,
                     [=](sycl::nd_item<1> item) { kernel_reset_followers(d_followers_kernel, num_points, item); });
  });

  gridSize = std::ceil(LayerTilesConstants::nRows * LayerTilesConstants::nColumns / static_cast<float>(blockSize[0]));
  workDiv = sycl::nd_range<1>(gridSize * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) { kernel_reset_hist(d_hist_kernel, item); });
  });
}

void CLUEAlgoSYCL::makeClusters(PointsCloud const &host_pc, PointsCloudSYCL &d_points, sycl::queue &stream) {
  setup(host_pc, d_points, stream);
  const int numThreadsPerBlock = 1024;  // ThreadsPerBlock = work-group size
  const sycl::range<1> blockSize(numThreadsPerBlock);
  const sycl::range<1> gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize[0])));
  PointsCloudSYCL::PointsCloudSYCLView *d_points_view = d_points.view();

  stream.submit([&](sycl::handler &cgh) {
    //SYCL kernels cannot capture by reference - need to reassign pointers inside the submit to pass by value
    auto d_hist_kernel = d_hist.get();
    auto num_points = host_pc.x.size();
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_compute_histogram(d_hist_kernel, d_points_view, num_points, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    auto dc_kernel = dc_;
    auto num_points = host_pc.x.size();
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_calculate_density(d_hist_kernel, d_points_view, dc_kernel, num_points, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    auto outlierDeltaFactor_kernel = outlierDeltaFactor_;
    auto dc_kernel = dc_;
    auto num_points = host_pc.x.size();
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_calculate_distanceToHigher(
          d_hist_kernel, d_points_view, outlierDeltaFactor_kernel, dc_kernel, num_points, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds.get();
    auto d_followers_kernel = d_followers.get();
    auto outlierDeltaFactor_kernel = outlierDeltaFactor_;
    auto dc_kernel = dc_;
    auto rhoc_kernel = rhoc_;
    auto num_points = host_pc.x.size();
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_find_clusters(d_seeds_kernel,
                           d_followers_kernel,
                           d_points_view,
                           outlierDeltaFactor_kernel,
                           dc_kernel,
                           rhoc_kernel,
                           num_points,
                           item);
    });
  });

  const sycl::range<1> gridSize_nseeds(ceil(maxNSeeds / static_cast<double>(blockSize[0])));
  stream.submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds.get();
    auto d_followers_kernel = d_followers.get();
    cgh.parallel_for(sycl::nd_range<1>(gridSize_nseeds * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_assign_clusters(d_seeds_kernel, d_followers_kernel, d_points_view, item);
    });
  });
  stream.wait();
}