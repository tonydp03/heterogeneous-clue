#ifndef Cluster_Collection_Alpaka_h
#define Cluster_Collection_Alpaka_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "DataFormats/ClusterCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr unsigned int reserve = 100000;

  class ClusterCollectionAlpaka {
  public:
    ClusterCollectionAlpaka() = delete;
    explicit ClusterCollectionAlpaka(Queue &stream)
        //input variables
        : x{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          y{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          z{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          eta{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          phi{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          r_over_absz{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          radius{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          layer{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          energy{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          isSilicon{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          //result variables
          rho{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          delta{cms::alpakatools::make_device_buffer<std::pair<float, int>[]>(stream, reserve)},
          nearestHigher{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          isSeed{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          tracksterIndex{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          view_d{cms::alpakatools::make_device_buffer<ClusterCollectionAlpakaView>(stream)} {
      auto view_h = cms::alpakatools::make_host_buffer<ClusterCollectionAlpakaView>(stream);
      view_h->x = x.data();
      view_h->y = y.data();
      view_h->z = z.data();
      view_h->eta = eta.data();
      view_h->phi = phi.data();
      view_h->r_over_absz = r_over_absz.data();
      view_h->radius = radius.data();
      view_h->layer = layer.data();
      view_h->energy = energy.data();
      view_h->isSilicon = isSilicon.data();
      view_h->rho = rho.data();
      view_h->delta = delta.data();
      view_h->nearestHigher = nearestHigher.data();
      view_h->isSeed = isSeed.data();
      view_h->tracksterIndex = tracksterIndex.data();

      alpaka::memcpy(stream, view_d, view_h);
      alpaka::wait(stream);
    }
    ClusterCollectionAlpaka(ClusterCollectionAlpaka const &) = delete;
    ClusterCollectionAlpaka(ClusterCollectionAlpaka &&) = default;
    ClusterCollectionAlpaka &operator=(ClusterCollectionAlpaka const &) = delete;
    ClusterCollectionAlpaka &operator=(ClusterCollectionAlpaka &&) = default;

    ~ClusterCollectionAlpaka() = default;

    cms::alpakatools::device_buffer<Device, float[]> x;
    cms::alpakatools::device_buffer<Device, float[]> y;
    cms::alpakatools::device_buffer<Device, float[]> z;
    cms::alpakatools::device_buffer<Device, float[]> eta;
    cms::alpakatools::device_buffer<Device, float[]> phi;
    cms::alpakatools::device_buffer<Device, float[]> r_over_absz;
    cms::alpakatools::device_buffer<Device, float[]> radius;
    cms::alpakatools::device_buffer<Device, int[]> layer;
    cms::alpakatools::device_buffer<Device, float[]> energy;
    cms::alpakatools::device_buffer<Device, int[]> isSilicon;
    cms::alpakatools::device_buffer<Device, float[]> rho;
    cms::alpakatools::device_buffer<Device, std::pair<float, int>[]> delta;
    cms::alpakatools::device_buffer<Device, int[]> nearestHigher;
    cms::alpakatools::device_buffer<Device, int[]> isSeed;
    cms::alpakatools::device_buffer<Device, int[]> tracksterIndex;

    class ClusterCollectionAlpakaView {
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

    ClusterCollectionAlpakaView *view() { return view_d.data(); }

  private:
    cms::alpakatools::device_buffer<Device, ClusterCollectionAlpakaView> view_d;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
