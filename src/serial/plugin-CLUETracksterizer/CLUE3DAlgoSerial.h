#ifndef CLUE3DAlgo_Serial_h
#define CLUE3DAlgo_Serial_h

#include "DataFormats/ClusterCollection.h"
#include "DataFormats/TICLLayerTile.h"

class CLUE3DAlgoSerial {
public:
  // constructor
  CLUE3DAlgoSerial() {
    hist_ = new TICLLayerTiles;
    histSoA_ = new TICLLayerTiles;
  }

  ~CLUE3DAlgoSerial() {
    delete hist_;
    delete histSoA_;
  };

  void makeTrackstersSoA(ClusterCollection const &host_pc, ClusterCollectionSerial &d_clustersSoA);
  void makeTracksters(ClusterCollection const &host_pc, ClusterCollectionSerialOnLayers &d_clusters);

  TICLLayerTiles *hist_;
  TICLLayerTiles *histSoA_;

private:
  void setupSoA(ClusterCollection const &pc, ClusterCollectionSerial &d_clustersSoA);
  void setup(ClusterCollection const &pc, ClusterCollectionSerialOnLayers &d_clusters);
};

#endif
