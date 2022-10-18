#include "DataFormats/PointsCloud.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "CLUEAlgoAlpaka.h"
#include "CLUEAlgoKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CLUEAlgoAlpaka::init_device() {
    d_hist = cms::alpakatools::make_device_buffer<LayerTilesAlpaka<Acc1D>[]>(queue_, NLAYERS);
    d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, maxNSeeds>>(queue_);
    d_followers =
        cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, maxNFollowers>[]>(queue_, reserve);

    hist_ = (*d_hist).data();
    seeds_ = (*d_seeds).data();
    followers_ = (*d_followers).data();
  }

  void CLUEAlgoAlpaka::setup(PointsCloud const &host_pc) {
    // copy input variables
    alpaka::memcpy(queue_, d_points.x, cms::alpakatools::make_host_view(host_pc.x.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.y, cms::alpakatools::make_host_view(host_pc.y.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.layer, cms::alpakatools::make_host_view(host_pc.layer.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(host_pc.weight.data(), host_pc.n));
    // initialize result and internal variables
    alpaka::memset(queue_, d_points.rho, 0x00, host_pc.n);
    alpaka::memset(queue_, d_points.delta, 0x00, host_pc.n);
    alpaka::memset(queue_, d_points.nearestHigher, 0x00, host_pc.n);
    alpaka::memset(queue_, d_points.clusterIndex, 0x00, host_pc.n);
    alpaka::memset(queue_, d_points.isSeed, 0x00, host_pc.n);
    alpaka::memset(queue_, (*d_hist), 0x00, static_cast<uint32_t>(NLAYERS));
    alpaka::memset(queue_, (*d_seeds), 0x00);
    alpaka::memset(queue_, (*d_followers), 0x00, host_pc.n);

    alpaka::wait(queue_);
  }

  void CLUEAlgoAlpaka::makeClusters(PointsCloud const &host_pc) {
    setup(host_pc);
    // calculate rho, delta and find seeds
    // 1 point per thread
    const Idx blockSize = 1024;
    const Idx gridSize = ceil(host_pc.n / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(WorkDiv1D, kernel_compute_histogram(), hist_, d_points.view(), d_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, kernel_calculate_density(), hist_, d_points.view(), dc_, d_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D,
                                                    kernel_calculate_distanceToHigher(),
                                                    hist_,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    d_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D,
                                                    kernel_find_clusters(),
                                                    seeds_,
                                                    followers_,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    rhoc_,
                                                    d_points.n));

    const Idx gridSize_nseeds = ceil(maxNSeeds / static_cast<float>(blockSize));
    auto WorkDiv1D_seeds = cms::alpakatools::make_workdiv<Acc1D>(gridSize_nseeds, blockSize);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D_seeds, kernel_assign_clusters(), seeds_, followers_, d_points.view()));

    alpaka::wait(queue_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
