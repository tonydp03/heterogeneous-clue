#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/ClusterCollection.h"
#include "DataFormats/CLUE_config.h"
#include "CLUE3DAlgoSerial.h"

class CLUESerialTracksterizer : public edm::EDProducer {
public:
  explicit CLUESerialTracksterizer(edm::ProductRegistry& reg);
  ~CLUESerialTracksterizer() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<ClusterCollection> clusterCollectionToken_;
  edm::EDPutTokenT<ClusterCollectionSerialOnLayers> tracksterToken_;
};

CLUESerialTracksterizer::CLUESerialTracksterizer(edm::ProductRegistry& reg)
    : clusterCollectionToken_{reg.consumes<ClusterCollection>()},
      tracksterToken_{reg.produces<ClusterCollectionSerialOnLayers>()} {}

void CLUESerialTracksterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  auto const& pc = event.get(clusterCollectionToken_);
  Parameters const& par = eventSetup.get<Parameters>();                         // new set of parameter?
  CLUE3DAlgoSerial clue3DAlgo(par.dc, par.rhoc, par.outlierDeltaFactor);
  clue3DAlgo.makeTracksters(pc);

  event.emplace(tracksterToken_, std::move(clue3DAlgo.d_clusters));
}

DEFINE_FWK_MODULE(CLUESerialTracksterizer);
