#ifndef CLUEAlgo_Serial_h
#define CLUEAlgo_Serial_h

#include "DataFormats/PointsCloud.h"
#include "DataFormats/LayerTilesSerial.h"

class CLUEAlgoSerial {
public:
  // constructor
  CLUEAlgoSerial() { hist_ = new std::array<LayerTilesSerial, NLAYERS>; };
  ~CLUEAlgoSerial() { delete hist_; };

  void makeClusters(PointsCloud const &host_pc, PointsCloudSerial &d_points, float const &dc, float const &rhoc, float const &outlierDeltaFactor);

  std::array<LayerTilesSerial, NLAYERS> *hist_;

private:
  void setup(PointsCloud const &host_pc, PointsCloudSerial &d_points);
};

#endif