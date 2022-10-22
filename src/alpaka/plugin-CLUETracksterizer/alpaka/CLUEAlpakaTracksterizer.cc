#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "DataFormats/ClusterCollection.h"
#include "AlpakaDataFormats/alpaka/ClusterCollectionAlpaka.h"
#include "CLUE3DAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEAlpakaTracksterizer : public edm::EDProducer {
  public:
    explicit CLUEAlpakaTracksterizer(edm::ProductRegistry& reg);
    ~CLUEAlpakaTracksterizer() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<ClusterCollection> clusterCollectionToken_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, ClusterCollectionAlpaka>> tracksterToken_;
    std::unique_ptr<CLUE3DAlgoAlpaka> algo_;
  };

  CLUEAlpakaTracksterizer::CLUEAlpakaTracksterizer(edm::ProductRegistry& reg)
      : clusterCollectionToken_{reg.consumes<ClusterCollection>()},
        tracksterToken_{reg.produces<cms::alpakatools::Product<Queue, ClusterCollectionAlpaka>>()} {}

  void CLUEAlpakaTracksterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
    auto const& pc = event.get(clusterCollectionToken_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx(event.streamID());
    auto stream = ctx.stream();
    // CLUE3DAlgoAlpaka algo_(stream);
    // algo_.makeTracksters(pc);
    // ctx.emplace(event, tracksterToken_, std::move(algo_.d_clusters));
    if (!algo_) {
      constexpr int reserve = 100000;
      algo_ = std::make_unique<CLUE3DAlgoAlpaka>(reserve, stream);
    }
    algo_->makeTracksters(pc);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEAlpakaTracksterizer);
