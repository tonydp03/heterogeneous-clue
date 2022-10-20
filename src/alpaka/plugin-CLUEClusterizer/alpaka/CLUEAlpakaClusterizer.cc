#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"
#include "AlpakaDataFormats/alpaka/PointsCloudAlpaka.h"
#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEAlpakaClusterizer : public edm::EDProducer {
  public:
    explicit CLUEAlpakaClusterizer(edm::ProductRegistry& reg);
    ~CLUEAlpakaClusterizer() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, PointsCloudAlpaka>> clusterToken_;
    std::unique_ptr<CLUEAlgoAlpaka> clueAlgo;
  };

  CLUEAlpakaClusterizer::CLUEAlpakaClusterizer(edm::ProductRegistry& reg)
      : pointsCloudToken_{reg.consumes<PointsCloud>()},
        clusterToken_{reg.produces<cms::alpakatools::Product<Queue, PointsCloudAlpaka>>()} {}

  void CLUEAlpakaClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
    auto const& pc = event.get(pointsCloudToken_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx(event.streamID());
    Parameters const& par = eventSetup.get<Parameters>();
    auto stream = ctx.stream();
    PointsCloudAlpaka d_pc{stream, static_cast<int>(pc.x.size())};
    if (!clueAlgo)
      clueAlgo = std::make_unique<CLUEAlgoAlpaka>(stream, par.dc, par.rhoc, par.outlierDeltaFactor);
    // CLUEAlgoAlpaka clueAlgo(pc.x.size(), par.dc, par.rhoc, par.outlierDeltaFactor, stream);
    // clueAlgo.makeClusters(pc, d_pc);
    clueAlgo->makeClusters(pc, d_pc);

    // ctx.emplace(event, clusterToken_, std::move(clueAlgo.d_points));
    ctx.emplace(event, clusterToken_, std::move(d_pc));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEAlpakaClusterizer);