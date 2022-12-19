#include "DataFormats/ClusterCollection.h"

#include "CLUE3DAlgoSYCL.h"
#include "CLUE3DAlgoKernels.h"
#include "DataFormats/Common.h"

#include "SYCLCore/device_unique_ptr.h"

constexpr int reserve = 100000;

CLUE3DAlgoSYCL::CLUE3DAlgoSYCL(sycl::queue const &stream)
    : d_hist{cms::sycltools::make_device_unique_uninitialized<TICLLayerTilesSYCL[]>(ticl::TileConstants::nLayers,
                                                                                    stream)},
      d_seeds{cms::sycltools::make_device_unique_uninitialized<cms::sycltools::VecArray<int, ticl::maxNSeeds>>(stream)},
      d_followers{
          cms::sycltools::make_device_unique_uninitialized<cms::sycltools::VecArray<int, ticl::maxNFollowers>[]>(
              reserve, stream)} {}

void CLUE3DAlgoSYCL::setup(ClusterCollection const &host_pc, ClusterCollectionSYCL &d_clusters, sycl::queue stream) {
  // copy input variables
  stream.memcpy(d_clusters.x.get(), host_pc.x.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.y.get(), host_pc.y.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.z.get(), host_pc.z.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.eta.get(), host_pc.eta.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.phi.get(), host_pc.phi.data(), host_pc.x.size() * sizeof(float));

  stream.memcpy(d_clusters.r_over_absz.get(), host_pc.r_over_absz.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.radius.get(), host_pc.radius.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.layer.get(), host_pc.layer.data(), host_pc.x.size() * sizeof(int));
  stream.memcpy(d_clusters.energy.get(), host_pc.energy.data(), host_pc.x.size() * sizeof(float));
  stream.memcpy(d_clusters.isSilicon.get(), host_pc.isSilicon.data(), host_pc.x.size() * sizeof(int));
  // initialize result and internal variables
  stream.memset(d_seeds.get(), 0x00, sizeof(cms::sycltools::VecArray<int, ticl::maxNSeeds>));

  const sycl::range<1> blockSize(1024);
  sycl::range<1> gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize[0])));
  auto workDiv = sycl::nd_range<1>(gridSize * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_followers_kernel = d_followers.get();
    auto num_points = host_pc.x.size();
    cgh.parallel_for(workDiv,
                     [=](sycl::nd_item<1> item) { kernel_reset_followers(d_followers_kernel, num_points, item); });
  });

  gridSize = std::ceil(ticl::TileConstants::nBins / static_cast<float>(blockSize[0]));
  workDiv = sycl::nd_range<1>(gridSize * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) { kernel_reset_hist(d_hist_kernel, item); });
  });
}

void CLUE3DAlgoSYCL::makeTracksters(ClusterCollection const &host_pc,
                                    ClusterCollectionSYCL &d_clusters,
                                    sycl::queue stream) {
  setup(host_pc, d_clusters, stream);

  // calculate rho, delta and find seeds
  // 1 point per thread
  const sycl::range<1> blockSize(1024);
  const sycl::range<1> gridSize(ceil(host_pc.x.size() / static_cast<float>(blockSize[0])));
  const auto workDiv = sycl::nd_range<1>(gridSize * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    auto d_clusters_kernel = d_clusters.view();
    auto numberOfPoints = host_pc.x.size();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) {
      kernel_compute_histogram(d_hist_kernel, d_clusters_kernel, numberOfPoints, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    auto d_clusters_kernel = d_clusters.view();
    auto numberOfPoints = host_pc.x.size();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) {
      kernel_calculate_density(d_hist_kernel, d_clusters_kernel, numberOfPoints, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist.get();
    auto d_clusters_kernel = d_clusters.view();
    auto numberOfPoints = host_pc.x.size();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) {
      kernel_compute_distance_to_higher(d_hist_kernel, d_clusters_kernel, numberOfPoints, item);
    });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds.get();
    auto d_followers_kernel = d_followers.get();
    auto d_clusters_kernel = d_clusters.view();
    auto numberOfPoints = host_pc.x.size();
    cgh.parallel_for(workDiv, [=](sycl::nd_item<1> item) {
      kernel_find_tracksters(d_seeds_kernel, d_followers_kernel, d_clusters_kernel, numberOfPoints, item);
    });
  });

  const sycl::range<1> gridSizeSeeds(ceil(ticl::maxNSeeds / static_cast<float>(blockSize[0])));
  const auto workDivSeeds = sycl::nd_range<1>(gridSizeSeeds * blockSize, blockSize);

  stream.submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds.get();
    auto d_followers_kernel = d_followers.get();
    auto d_clusters_kernel = d_clusters.view();
    cgh.parallel_for(workDivSeeds, [=](sycl::nd_item<1> item) {
      kernel_assign_tracksters(d_seeds_kernel, d_followers_kernel, d_clusters_kernel, item);
    });
  });

#ifdef TRACKSTER_DEBUG
  // print the number of Tracksters
  auto workDivPrint = sycl::nd_range<1>(1, 1);
  stream.submit([&](sycl::handler &cgh) {
    auto d_clusters_kernel = d_clusters.view();
    auto numberOfPoints = host_pc.x.size();
    sycl::stream out(1024, 768, cgh);
    cgh.parallel_for(workDivPrint, [=](sycl::nd_item<1> item) {
      kernel_print_ntracksters(d_clusters_kernel, numberOfPoints, item, out);
    });
  });
#endif

  stream.wait();
}
