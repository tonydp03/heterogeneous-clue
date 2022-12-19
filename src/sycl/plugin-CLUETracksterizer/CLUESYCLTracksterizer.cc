#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/Product.h"

#include "DataFormats/ClusterCollection.h"
#include "SYCLDataFormats/ClusterCollectionSYCL.h"
#include "CLUE3DAlgoSYCL.h"



  class CLUESYCLTracksterizer : public edm::EDProducer {
  public:
    explicit CLUESYCLTracksterizer(edm::ProductRegistry& reg);
    ~CLUESYCLTracksterizer() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<ClusterCollection> clusterCollectionToken_;
    edm::EDPutTokenT<cms::sycltools::Product<ClusterCollectionSYCL>> tracksterToken_;
    std::unique_ptr<CLUE3DAlgoSYCL> algo_;
  };

  CLUESYCLTracksterizer::CLUESYCLTracksterizer(edm::ProductRegistry& reg)
      : clusterCollectionToken_{reg.consumes<ClusterCollection>()},
        tracksterToken_{reg.produces<cms::sycltools::Product<ClusterCollectionSYCL>>()} {}

  void CLUESYCLTracksterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
    auto const& pc = event.get(clusterCollectionToken_);
    cms::sycltools::ScopedContextProduce ctx(event.streamID());
    auto stream = ctx.stream();
    ClusterCollectionSYCL d_clusters(stream, pc.x.size());
    if (!algo_)
      algo_ = std::make_unique<CLUE3DAlgoSYCL>(stream);
    algo_->makeTracksters(pc, d_clusters, stream);
    ctx.emplace(event, tracksterToken_, std::move(d_clusters));
  }


DEFINE_FWK_MODULE(CLUESYCLTracksterizer);
