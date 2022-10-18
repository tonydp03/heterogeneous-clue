#ifndef CLUEAlgo_CUDA_h
#define CLUEAlgo_CUDA_h

// #include <optional>

#include "CUDACore/device_unique_ptr.h"
#include "CUDADataFormats/PointsCloudCUDA.h"
#include "CUDADataFormats/LayerTilesCUDA.h"

class CLUEAlgoCUDA {
public:
  // constructor
  CLUEAlgoCUDA() = delete;
  explicit CLUEAlgoCUDA(float const &dc, float const &rhoc, float const &outlierDeltaFactor, cudaStream_t stream)
      : d_points{stream}, dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor}, stream_{stream} {
    init_device();
  }

  ~CLUEAlgoCUDA() = default;

  void makeClusters(PointsCloud const &host_pc);

  PointsCloudCUDA d_points;

  LayerTilesCUDA *hist_;
  cms::cuda::VecArray<int, maxNSeeds> *seeds_;
  cms::cuda::VecArray<int, maxNFollowers> *followers_;

private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;
  cudaStream_t stream_ = nullptr;
  cms::cuda::device::unique_ptr<LayerTilesCUDA[]> d_hist;
  cms::cuda::device::unique_ptr<cms::cuda::VecArray<int, maxNSeeds>> d_seeds;
  cms::cuda::device::unique_ptr<cms::cuda::VecArray<int, maxNFollowers>[]> d_followers;

  // private methods
  void init_device();

  void setup(PointsCloud const &host_pc);
};
#endif
