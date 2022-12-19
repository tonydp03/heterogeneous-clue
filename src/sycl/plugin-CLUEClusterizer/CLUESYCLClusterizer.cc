#include <CL/sycl.hpp>

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/Product.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"
#include "SYCLDataFormats/PointsCloudSYCL.h"
#include "CLUEAlgoSYCL.h"

class CLUESYCLClusterizer : public edm::EDProducer {
public:
  explicit CLUESYCLClusterizer(edm::ProductRegistry& reg);
  ~CLUESYCLClusterizer() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
  edm::EDPutTokenT<cms::sycltools::Product<PointsCloudSYCL>> clusterToken_;
  std::unique_ptr<CLUEAlgoSYCL> clueAlgo;
};

CLUESYCLClusterizer::CLUESYCLClusterizer(edm::ProductRegistry& reg)
    : pointsCloudToken_{reg.consumes<PointsCloud>()},
      clusterToken_{reg.produces<cms::sycltools::Product<PointsCloudSYCL>>()} {}

void CLUESYCLClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  auto const& pc = event.get(pointsCloudToken_);
  cms::sycltools::ScopedContextProduce ctx(event.streamID());
  Parameters const& par = eventSetup.get<Parameters>();
  auto stream = ctx.stream();
  PointsCloudSYCL d_points(stream, pc.x.size());
  if (!clueAlgo) {
    clueAlgo = std::make_unique<CLUEAlgoSYCL>(par.dc, par.rhoc, par.outlierDeltaFactor, stream);
  }
  clueAlgo->makeClusters(pc, d_points, stream);

  ctx.emplace(event, clusterToken_, std::move(d_points));
}

DEFINE_FWK_MODULE(CLUESYCLClusterizer);