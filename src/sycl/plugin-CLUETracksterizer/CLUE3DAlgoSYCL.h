#ifndef CLUE3DAlgo_SYCL_h
#define CLUE3DAlgo_SYCL_h

#include <CL/sycl.hpp>

#include "SYCLCore/device_unique_ptr.h"

#include "DataFormats/Common.h"
#include "SYCLDataFormats/TICLLayerTilesSYCL.h"
#include "SYCLDataFormats/ClusterCollectionSYCL.h"

class CLUE3DAlgoSYCL {
public:
  CLUE3DAlgoSYCL() = delete;
  CLUE3DAlgoSYCL(sycl::queue const &stream);
  ~CLUE3DAlgoSYCL() = default;

  void makeTracksters(ClusterCollection const &host_pc, ClusterCollectionSYCL &d_clusters, sycl::queue stream);

private:
  cms::sycltools::device::unique_ptr<TICLLayerTilesSYCL[]> d_hist;
  cms::sycltools::device::unique_ptr<cms::sycltools::VecArray<int, ticl::maxNSeeds>> d_seeds;
  cms::sycltools::device::unique_ptr<cms::sycltools::VecArray<int, ticl::maxNFollowers>[]> d_followers;

  void setup(ClusterCollection const &host_pc, ClusterCollectionSYCL &d_clusters, sycl::queue stream);
};

#endif