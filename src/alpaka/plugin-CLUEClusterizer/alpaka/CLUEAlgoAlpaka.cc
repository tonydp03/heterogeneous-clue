#include "DataFormats/PointsCloud.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "CLUEAlgoAlpaka.h"
#include "CLUEAlgoKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr int reserve = 1000000;

  void CLUEAlgoAlpaka::init_device(Queue queue_) {
    d_hist = cms::alpakatools::make_device_buffer<LayerTilesAlpaka[]>(queue_, NLAYERS);
    d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, maxNSeeds>>(queue_);
    d_followers =
        cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, maxNFollowers>[]>(queue_, reserve);
    hist_ = (*d_hist).data();
    seeds_ = (*d_seeds).data();
    followers_ = (*d_followers).data();
    const Idx blockSize = 1024;
    Idx gridSize = std::ceil(LayerTilesConstants::nRows * LayerTilesConstants::nColumns / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(queue_, alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelSetHistoPtrs(), hist_));

  }

  void CLUEAlgoAlpaka::setup(PointsCloud const &host_pc, PointsCloudAlpaka &d_points, Queue queue_) {
    // copy input variables
    alpaka::memcpy(queue_, d_points.x, cms::alpakatools::make_host_view(host_pc.x.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_points.y, cms::alpakatools::make_host_view(host_pc.y.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_points.layer, cms::alpakatools::make_host_view(host_pc.layer.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(host_pc.weight.data(), host_pc.x.size()));
    // initialize result and internal variables
    // alpaka::memset(queue_, d_points.rho, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    // alpaka::memset(queue_, d_points.delta, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    // alpaka::memset(queue_, d_points.nearestHigher, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    // alpaka::memset(queue_, d_points.clusterIndex, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    // alpaka::memset(queue_, d_points.isSeed, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    // alpaka::memset(queue_, (*d_hist), 0x00, static_cast<uint32_t>(NLAYERS));
    alpaka::memset(queue_, (*d_seeds), 0x00);
    // alpaka::memset(queue_, (*d_followers), 0x00, static_cast<uint32_t>(host_pc.x.size()));
    const Idx blockSize = 1024;
    Idx gridSize = std::ceil(host_pc.x.size() / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelResetFollowers(), followers_, host_pc.x.size()));

  }

  void CLUEAlgoAlpaka::makeClusters(PointsCloud const &host_pc, PointsCloudAlpaka &d_points, Queue queue_) {
    setup(host_pc, d_points, queue_);
    const Idx blockSize = 1024;
    Idx gridSize = std::ceil(LayerTilesConstants::nRows * LayerTilesConstants::nColumns / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(queue_, alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelResetHist(), hist_, d_points.view(), host_pc.x.size()));
    // calculate rho, delta and find seeds
    // 1 point per thread
    gridSize = ceil(host_pc.x.size() / static_cast<float>(blockSize));
    WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelComputeHistogram(), hist_, d_points.view(), host_pc.x.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, KernelCalculateDensity(), hist_, d_points.view(), dc_, host_pc.x.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D,
                                                    KernelComputeDistanceToHigher(),
                                                    hist_,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    host_pc.x.size())); 
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(WorkDiv1D,
                                                    KernelFindClusters(),
                                                    seeds_,
                                                    followers_,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    rhoc_,
                                                    host_pc.x.size()));

    const Idx gridSize_seeds = ceil(maxNSeeds / static_cast<float>(blockSize));
    auto WorkDiv1D_seeds = cms::alpakatools::make_workdiv<Acc1D>(gridSize_seeds, blockSize);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(WorkDiv1D_seeds, KernelAssignClusters(), seeds_, followers_, d_points.view()));
    alpaka::wait(queue_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
