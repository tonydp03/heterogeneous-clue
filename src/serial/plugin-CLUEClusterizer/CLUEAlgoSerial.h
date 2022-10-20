#ifndef CLUEAlgo_Serial_h
#define CLUEAlgo_Serial_h

#include "DataFormats/PointsCloud.h"
#include "DataFormats/LayerTilesSerial.h"

class CLUEAlgoSerial {
public:
  // constructor
  CLUEAlgoSerial() = delete;
  CLUEAlgoSerial(float const &dc, float const &rhoc, float const &outlierDeltaFactor)
      : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {
    hist_ = new LayerTilesSerial[NLAYERS];
    seeds = new cms::cuda::VecArray<int, 100000>;
    followers = new cms::cuda::VecArray<int, 128>[1000000];
  };
  ~CLUEAlgoSerial() {
    delete hist_;
    delete seeds;
    delete followers;
  };

  void makeClusters(PointsCloud const &host_pc, PointsCloudSerial &d_pc);

  // PointsCloudSerial d_points;

  // std::array<LayerTilesSerial, NLAYERS> hist_;
  // cms::cuda::VecArray<int, 128> seeds;
  // std::vector<cms::cuda::VecArray<int, 128>> followers;
  LayerTilesSerial *hist_;
  cms::cuda::VecArray<int, 100000> *seeds;
  cms::cuda::VecArray<int, 128> *followers;

private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  void setup(PointsCloud const &host_pc, PointsCloudSerial &d_pc);
};

#endif