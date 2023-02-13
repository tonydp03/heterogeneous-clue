#ifndef CLUE3DAlgo_Alpaka_h
#define CLUE3DAlgo_Alpaka_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/alpaka/ClusterCollectionAlpaka.h"
#include "AlpakaDataFormats/TICLLayerTilesAlpaka.h"
#include "AlpakaDataFormats/AlpakaVecArray.h"

#include "DataFormats/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUE3DAlgoAlpaka {
  public:
    // constructor
    CLUE3DAlgoAlpaka() = delete;
    explicit CLUE3DAlgoAlpaka(Queue stream) { init_device(stream); }

    ~CLUE3DAlgoAlpaka() = default;

    void makeTracksters(ClusterCollection const &host_pc, ClusterCollectionAlpaka &d_clusters, Queue stream);

    TICLLayerTilesAlpaka *hist_;
    cms::alpakatools::VecArray<int, ticl::maxNSeeds> *seeds_;
    cms::alpakatools::VecArray<int, ticl::maxNFollowers> *followers_;

  private:
    std::optional<cms::alpakatools::device_buffer<Device, TICLLayerTilesAlpaka[]>> d_hist;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int, ticl::maxNSeeds>>> d_seeds;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int, ticl::maxNFollowers>[]>>
        d_followers;

    void init_device(Queue stream);

    void setup(ClusterCollection const &host_pc, ClusterCollectionAlpaka &d_clusters, Queue stream);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
