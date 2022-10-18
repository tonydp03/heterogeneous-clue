#include "DataFormats/ClusterCollection.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "CLUE3DAlgoAlpaka.h"
#include "CLUE3DAlgoKernels.h"
#include "DataFormats/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CLUE3DAlgoAlpaka::init_device() {
    d_hist = cms::alpakatools::make_device_buffer<TICLLayerTilesAlpaka[]>(queue_, ticl::TileConstants::nLayers);
    d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, ticl::maxNSeeds>>(queue_);
    d_followers =
        cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int, ticl::maxNFollowers>[]>(queue_, reserve);

    hist_ = (*d_hist).data();
    seeds_ = (*d_seeds).data();
    followers_ = (*d_followers).data();
  }

  void CLUE3DAlgoAlpaka::setup(ClusterCollection const &host_pc) {
    // copy input variables
    alpaka::memcpy(queue_, d_clusters.x, cms::alpakatools::make_host_view(host_pc.x.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_clusters.y, cms::alpakatools::make_host_view(host_pc.y.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_clusters.z, cms::alpakatools::make_host_view(host_pc.z.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_clusters.eta, cms::alpakatools::make_host_view(host_pc.eta.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_clusters.phi, cms::alpakatools::make_host_view(host_pc.phi.data(), host_pc.x.size()));
    alpaka::memcpy(
        queue_, d_clusters.r_over_absz, cms::alpakatools::make_host_view(host_pc.r_over_absz.data(), host_pc.x.size()));
    alpaka::memcpy(
        queue_, d_clusters.radius, cms::alpakatools::make_host_view(host_pc.radius.data(), host_pc.x.size()));
    alpaka::memcpy(queue_, d_clusters.layer, cms::alpakatools::make_host_view(host_pc.layer.data(), host_pc.x.size()));
    alpaka::memcpy(
        queue_, d_clusters.energy, cms::alpakatools::make_host_view(host_pc.energy.data(), host_pc.x.size()));
    alpaka::memcpy(
        queue_, d_clusters.isSilicon, cms::alpakatools::make_host_view(host_pc.isSilicon.data(), host_pc.x.size()));
    // initialize result and internal variables
    alpaka::memset(queue_, d_clusters.rho, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    alpaka::memset(queue_, d_clusters.delta, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    alpaka::memset(queue_, d_clusters.nearestHigher, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    alpaka::memset(queue_, d_clusters.tracksterIndex, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    alpaka::memset(queue_, d_clusters.isSeed, 0x00, static_cast<uint32_t>(host_pc.x.size()));
    alpaka::memset(queue_, (*d_hist), 0x00, static_cast<uint32_t>(ticl::TileConstants::nLayers));
    alpaka::memset(queue_, (*d_seeds), 0x00);
    alpaka::memset(queue_, (*d_followers), 0x00, static_cast<uint32_t>(host_pc.x.size()));

    alpaka::wait(queue_);
  }

  void CLUE3DAlgoAlpaka::makeTracksters(ClusterCollection const &host_pc) {
    setup(host_pc);
    // calculate rho, delta and find seeds
    // 1 point per thread
    const Idx blockSize = 1024;
    const Idx gridSize = ceil(host_pc.x.size() / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, KernelComputeHistogram(), hist_, d_clusters.view(), host_pc.x.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, KernelCalculateDensity(), hist_, d_clusters.view(), host_pc.x.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, KernelComputeDistanceToHigher(), hist_, d_clusters.view(), host_pc.x.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D, KernelFindClusters(), seeds_, followers_, d_clusters.view(), host_pc.x.size()));

    const Idx gridSize_seeds = ceil(ticl::maxNSeeds / static_cast<float>(blockSize));
    auto WorkDiv1D_seeds = cms::alpakatools::make_workdiv<Acc1D>(gridSize_seeds, blockSize);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        WorkDiv1D_seeds, KernelAssignClusters(), seeds_, followers_, d_clusters.view()));

    // To validate the number of Tracksters
    // auto WorkDivPrint = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    // alpaka::enqueue(
    //     queue_,
    //     alpaka::createTaskKernel<Acc1D>(WorkDivPrint, KernelPrintNTracksters(), d_clusters.view(), host_pc.x.size()));

    alpaka::wait(queue_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE