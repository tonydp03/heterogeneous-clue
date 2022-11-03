#ifndef Points_Cloud_CUDA_h
#define Points_Cloud_CUDA_h

#include <memory>

#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/device_unique_ptr.h"
#include "DataFormats/PointsCloud.h"

class PointsCloudCUDA {
public:
  PointsCloudCUDA() = delete;
  explicit PointsCloudCUDA(cudaStream_t stream, int nPoints)
      // input variables
      : x{cms::cuda::make_device_unique<float[]>(nPoints, stream)},
        y{cms::cuda::make_device_unique<float[]>(nPoints, stream)},
        layer{cms::cuda::make_device_unique<int[]>(nPoints, stream)},
        weight{cms::cuda::make_device_unique<float[]>(nPoints, stream)},
        // result variables
        rho{cms::cuda::make_device_unique<float[]>(nPoints, stream)},
        delta{cms::cuda::make_device_unique<float[]>(nPoints, stream)},
        nearestHigher{cms::cuda::make_device_unique<int[]>(nPoints, stream)},
        clusterIndex{cms::cuda::make_device_unique<int[]>(nPoints, stream)},
        isSeed{cms::cuda::make_device_unique<int[]>(nPoints, stream)},
        view_d{cms::cuda::make_device_unique<PointsCloudCUDAView>(stream)} {
    auto view_h = cms::cuda::make_host_unique<PointsCloudCUDAView>(stream);
    view_h->x = x.get();
    view_h->y = y.get();
    view_h->layer = layer.get();
    view_h->weight = weight.get();
    view_h->rho = rho.get();
    view_h->delta = delta.get();
    view_h->nearestHigher = nearestHigher.get();
    view_h->clusterIndex = clusterIndex.get();
    view_h->isSeed = isSeed.get();

    cudaMemcpyAsync(view_d.get(), view_h.get(), sizeof(PointsCloudCUDAView), cudaMemcpyHostToDevice, stream);
  }
  PointsCloudCUDA(PointsCloudCUDA const &) = delete;
  PointsCloudCUDA(PointsCloudCUDA &&) = default;
  PointsCloudCUDA &operator=(PointsCloudCUDA const &) = delete;
  PointsCloudCUDA &operator=(PointsCloudCUDA &&) = default;

  ~PointsCloudCUDA() = default;

  cms::cuda::device::unique_ptr<float[]> x;
  cms::cuda::device::unique_ptr<float[]> y;
  cms::cuda::device::unique_ptr<int[]> layer;
  cms::cuda::device::unique_ptr<float[]> weight;
  cms::cuda::device::unique_ptr<float[]> rho;
  cms::cuda::device::unique_ptr<float[]> delta;
  cms::cuda::device::unique_ptr<int[]> nearestHigher;
  cms::cuda::device::unique_ptr<int[]> clusterIndex;
  cms::cuda::device::unique_ptr<int[]> isSeed;

  class PointsCloudCUDAView {
  public:
    float *x;
    float *y;
    int *layer;
    float *weight;
    float *rho;
    float *delta;
    int *nearestHigher;
    int *clusterIndex;
    int *isSeed;
  };

  PointsCloudCUDAView *view() { return view_d.get(); }

private:
  cms::cuda::device::unique_ptr<PointsCloudCUDAView> view_d;
};

#endif
