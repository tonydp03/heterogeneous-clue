#ifndef CLUEAlgo_Alpaka_h
#define CLUEAlgo_Alpaka_h

// #include <optional>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/alpaka/PointsCloudAlpaka.h"
#include "AlpakaDataFormats/LayerTilesAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEAlgoAlpaka {
  public:
    // constructor
    CLUEAlgoAlpaka() = delete;
    //  explicit CLUEAlgoAlpaka(int nPoints, float const &dc, float const &rhoc, float const &outlierDeltaFactor, Queue stream)
    //      : d_points{stream, nPoints}, queue_{std::move(stream)}, dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {

    explicit CLUEAlgoAlpaka(Queue stream, float const &dc, float const &rhoc, float const &outlierDeltaFactor)
    : queue_{std::move(stream)}, dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {
      init_device();
    }

    ~CLUEAlgoAlpaka() = default;

    void makeClusters(PointsCloud const &host_pc, PointsCloudAlpaka &d_points);

    // PointsCloudAlpaka d_points;

    LayerTilesAlpaka *hist_;
    cms::alpakatools::VecArray<int, maxNSeeds> *seeds_;
    cms::alpakatools::VecArray<int, maxNFollowers> *followers_;

  private:
    Queue queue_;
    float dc_;
    float rhoc_;
    float outlierDeltaFactor_;

    std::optional<cms::alpakatools::device_buffer<Device, LayerTilesAlpaka[]>> d_hist;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int, maxNSeeds>>> d_seeds;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int, maxNFollowers>[]>> d_followers;

    // private methods
    // void init_device(int nPoints);
    void init_device();

    void setup(PointsCloud const &host_pc, PointsCloudAlpaka &d_points);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
