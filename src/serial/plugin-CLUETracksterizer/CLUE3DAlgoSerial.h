#ifndef CLUE3DAlgo_Serial_h
#define CLUE3DAlgo_Serial_h

#include "DataFormats/ClusterCollection.h"
#include "DataFormats/TICLLayerTile.h"

class CLUE3DAlgoSerial {
public:
  // constructor
  CLUE3DAlgoSerial() {
      hist_ = new TICLLayerTiles;
  }

  ~CLUE3DAlgoSerial() {delete hist_;};

  void makeTracksters(ClusterCollection const &host_pc);

  ClusterCollectionSerialOnLayers d_clusters;

  TICLLayerTiles *hist_;

private:

  void setup(ClusterCollection const &pc);
};

#endif
