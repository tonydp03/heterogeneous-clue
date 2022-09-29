#ifndef CLUE3DAlgo_Serial_h
#define CLUE3DAlgo_Serial_h

#include "DataFormats/ClusterCollection.h"
#include "DataFormats/TICLLayerTile.h"

class CLUE3DAlgoSerial {
public:
  // constructor
  CLUE3DAlgoSerial() = delete;
  // set the right parameters in the constructor
  explicit CLUE3DAlgoSerial(float const &dc,
                            float const &rhoc,
                            float const &outlierDeltaFactor)
      : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {
      hist_ = new TICLLayerTiles;
      }

  ~CLUE3DAlgoSerial() = default;

  void makeTracksters(ClusterCollection const &host_pc);

  ClusterCollectionSerialOnLayers d_clusters;

  TICLLayerTiles *hist_;

private:
  // parameters needed for 3D?
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  void setup(ClusterCollection const &pc);
};

#endif
