#ifndef CLUE3DAlgo_Serial_h
#define CLUE3DAlgo_Serial_h

#include "DataFormats/PointsCloud.h"
#include "DataFormats/ClusterCollection.h"
#include "DataFormats/LayerTilesSerial.h"

class CLUE3DAlgoSerial {
public:
  // constructor
  CLUE3DAlgoSerial() = delete;
  // set the right parameters in the constructor
  explicit CLUE3DAlgoSerial(float const &dc,
                            float const &rhoc,
                            float const &outlierDeltaFactor,
                            uint32_t const &numberOfPoints)
      : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {}

  ~CLUE3DAlgoSerial() = default;

  void makeTracksters(PointsCloudSerial const &pc);

  ClusterCollection d_clusters;

  std::array<LayerTilesSerial, NLAYERS>
      hist_;  // ?????? maybe same layer tiles but with different size and other functions?

private:
  // parameters needed for 3D?
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  void setup(PointsCloudSerial const &pc);
};

#endif