#ifndef Cluster_Collection_SYCL_h
#define Cluster_Collection_SYCL_h

#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/device_unique_ptr.h"
#include "DataFormats/ClusterCollection.h"

class ClusterCollectionSYCL {
public:
  ClusterCollectionSYCL() = delete;
  explicit ClusterCollectionSYCL(sycl::queue stream, int nPoints)
      //input variables
      : x{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        y{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        z{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        eta{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        phi{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        r_over_absz{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        radius{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        layer{cms::sycltools::make_device_unique<int[]>(nPoints, stream)},
        energy{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        isSilicon{cms::sycltools::make_device_unique<int[]>(nPoints, stream)},
        //result variables
        rho{cms::sycltools::make_device_unique<float[]>(nPoints, stream)},
        delta{cms::sycltools::make_device_unique_uninitialized<std::pair<float, int>[]>(nPoints, stream)},
        nearestHigher{cms::sycltools::make_device_unique<int[]>(nPoints, stream)},
        isSeed{cms::sycltools::make_device_unique<int[]>(nPoints, stream)},
        tracksterIndex{cms::sycltools::make_device_unique<int[]>(nPoints, stream)},
        view_d{cms::sycltools::make_device_unique<ClusterCollectionSYCLView>(stream)} {
    auto view_h = cms::sycltools::make_host_unique<ClusterCollectionSYCLView>(stream);
    view_h->x = x.get();
    view_h->y = y.get();
    view_h->z = z.get();
    view_h->eta = eta.get();
    view_h->phi = phi.get();
    view_h->r_over_absz = r_over_absz.get();
    view_h->radius = radius.get();
    view_h->layer = layer.get();
    view_h->energy = energy.get();
    view_h->isSilicon = isSilicon.get();
    view_h->rho = rho.get();
    view_h->delta = delta.get();
    view_h->nearestHigher = nearestHigher.get();
    view_h->isSeed = isSeed.get();
    view_h->tracksterIndex = tracksterIndex.get();

    stream.memcpy(view_d.get(), view_h.get(), sizeof(ClusterCollectionSYCLView)).wait();
  }
  ClusterCollectionSYCL(ClusterCollectionSYCL const &) = delete;
  ClusterCollectionSYCL(ClusterCollectionSYCL &&) = default;
  ClusterCollectionSYCL &operator=(ClusterCollectionSYCL const &) = delete;
  ClusterCollectionSYCL &operator=(ClusterCollectionSYCL &&) = default;

  ~ClusterCollectionSYCL() = default;

  cms::sycltools::device::unique_ptr<float[]> x;
  cms::sycltools::device::unique_ptr<float[]> y;
  cms::sycltools::device::unique_ptr<float[]> z;
  cms::sycltools::device::unique_ptr<float[]> eta;
  cms::sycltools::device::unique_ptr<float[]> phi;
  cms::sycltools::device::unique_ptr<float[]> r_over_absz;
  cms::sycltools::device::unique_ptr<float[]> radius;
  cms::sycltools::device::unique_ptr<int[]> layer;
  cms::sycltools::device::unique_ptr<float[]> energy;
  cms::sycltools::device::unique_ptr<int[]> isSilicon;
  cms::sycltools::device::unique_ptr<float[]> rho;
  cms::sycltools::device::unique_ptr<std::pair<float, int>[]> delta;
  cms::sycltools::device::unique_ptr<int[]> nearestHigher;
  cms::sycltools::device::unique_ptr<int[]> isSeed;
  cms::sycltools::device::unique_ptr<int[]> tracksterIndex;

  class ClusterCollectionSYCLView {
  public:
    float *x;
    float *y;
    float *z;
    float *eta;
    float *phi;
    float *r_over_absz;
    float *radius;
    int *layer;
    float *energy;
    int *isSilicon;
    float *rho;
    std::pair<float, int> *delta;
    int *nearestHigher;
    int *isSeed;
    int *tracksterIndex;
  };

  ClusterCollectionSYCLView *view() { return view_d.get(); }

private:
  cms::sycltools::device::unique_ptr<ClusterCollectionSYCLView> view_d;
};

#endif
