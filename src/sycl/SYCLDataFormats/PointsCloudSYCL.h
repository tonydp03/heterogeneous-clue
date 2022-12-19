#ifndef POINTS_CLOUD_SYCL_H
#define POINTS_CLOUD_SYCL_H

#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/device_unique_ptr.h"

constexpr unsigned int reserve = 1000000;

class PointsCloudSYCL {
public:
  PointsCloudSYCL() = default;
  PointsCloudSYCL(sycl::queue stream, int numberOfPoints)
      : x{cms::sycltools::make_device_unique<float[]>(reserve, stream)},
        y{cms::sycltools::make_device_unique<float[]>(reserve, stream)},
        layer{cms::sycltools::make_device_unique<int[]>(reserve, stream)},
        weight{cms::sycltools::make_device_unique<float[]>(reserve, stream)},
        rho{cms::sycltools::make_device_unique<float[]>(reserve, stream)},
        delta{cms::sycltools::make_device_unique<float[]>(reserve, stream)},
        nearestHigher{cms::sycltools::make_device_unique<int[]>(reserve, stream)},
        clusterIndex{cms::sycltools::make_device_unique<int[]>(reserve, stream)},
        isSeed{cms::sycltools::make_device_unique<int[]>(reserve, stream)},
        n{numberOfPoints} {
    auto view = cms::sycltools::make_host_unique<PointsCloudSYCLView>(stream);
    view->x = x.get();
    view->y = y.get();
    view->layer = layer.get();
    view->weight = weight.get();
    view->rho = rho.get();
    view->delta = delta.get();
    view->nearestHigher = nearestHigher.get();
    view->clusterIndex = clusterIndex.get();
    view->isSeed = isSeed.get();
    view->n = numberOfPoints;

    view_d = cms::sycltools::make_device_unique<PointsCloudSYCLView>(stream);
    stream.memcpy(view_d.get(), view.get(), sizeof(PointsCloudSYCLView)).wait();
  }
  PointsCloudSYCL(PointsCloudSYCL const &pc) = delete;
  PointsCloudSYCL(PointsCloudSYCL &&) = default;
  PointsCloudSYCL &operator=(PointsCloudSYCL const &) = delete;
  PointsCloudSYCL &operator=(PointsCloudSYCL &&) = default;

  ~PointsCloudSYCL() = default;

  cms::sycltools::device::unique_ptr<float[]> x;
  cms::sycltools::device::unique_ptr<float[]> y;
  cms::sycltools::device::unique_ptr<int[]> layer;
  cms::sycltools::device::unique_ptr<float[]> weight;
  cms::sycltools::device::unique_ptr<float[]> rho;
  cms::sycltools::device::unique_ptr<float[]> delta;
  cms::sycltools::device::unique_ptr<int[]> nearestHigher;
  cms::sycltools::device::unique_ptr<int[]> clusterIndex;
  cms::sycltools::device::unique_ptr<int[]> isSeed;
  int n;

  class PointsCloudSYCLView {
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
    int n;
  };

  PointsCloudSYCLView *view() const { return view_d.get(); }

private:
  cms::sycltools::device::unique_ptr<PointsCloudSYCLView> view_d;
};
#endif