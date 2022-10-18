#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "DataFormats/CLUE_config.h"
#include "DataFormats/PointsCloud.h"

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"

#include "AlpakaDataFormats/alpaka/PointsCloudAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEOutputProducer : public edm::EDProducer {
  public:
    explicit CLUEOutputProducer(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
    edm::EDGetTokenT<cms::alpakatools::Product<Queue, PointsCloudAlpaka>> deviceClustersToken_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, PointsCloud>> resultsToken_;
    edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
  };

  CLUEOutputProducer::CLUEOutputProducer(edm::ProductRegistry& reg)
      : deviceClustersToken_(reg.consumes<cms::alpakatools::Product<Queue, PointsCloudAlpaka>>()),
        resultsToken_(reg.produces<cms::alpakatools::Product<Queue, PointsCloud>>()),
        pointsCloudToken_(reg.consumes<PointsCloud>()) {}

  void CLUEOutputProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto outDir = eventSetup.get<std::filesystem::path>();
    auto results = event.get(pointsCloudToken_);
    auto const& pcProduct = event.get(deviceClustersToken_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx{pcProduct};
    auto const& device_clusters = ctx.get(pcProduct);
    auto stream = ctx.stream();

    results.outResize(device_clusters.n);
    alpaka::memcpy(stream,
                   cms::alpakatools::make_host_view(results.rho.data(), results.n),
                   device_clusters.rho,
                   device_clusters.n);
    alpaka::memcpy(stream,
                   cms::alpakatools::make_host_view(results.delta.data(), results.n),
                   device_clusters.delta,
                   device_clusters.n);
    alpaka::memcpy(stream,
                   cms::alpakatools::make_host_view(results.nearestHigher.data(), results.n),
                   device_clusters.nearestHigher,
                   device_clusters.n);
    alpaka::memcpy(stream,
                   cms::alpakatools::make_host_view(results.isSeed.data(), results.n),
                   device_clusters.isSeed,
                   device_clusters.n);
    alpaka::memcpy(stream,
                   cms::alpakatools::make_host_view(results.clusterIndex.data(), results.n),
                   device_clusters.clusterIndex,
                   device_clusters.n);
    alpaka::wait(stream);

    std::cout << "Data transferred back to host" << std::endl;

    Parameters par;
    par = eventSetup.get<Parameters>();
    if (par.produceOutput) {
      auto temp_outDir = eventSetup.get<std::filesystem::path>();
      std::string input_file_name = temp_outDir.filename();
      std::string output_file_name = create_outputfileName(input_file_name, par.dc, par.rhoc, par.outlierDeltaFactor);
      std::filesystem::path outDir = temp_outDir.parent_path() / output_file_name;

      std::ofstream clueOut(outDir);

      clueOut << "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
      for (unsigned int i = 0; i < device_clusters.n; i++) {
        clueOut << i << "," << results.x[i] << "," << results.y[i] << "," << results.layer[i] << ","
                << results.weight[i] << "," << results.rho[i] << ","
                << (results.delta[i] > 999 ? 999 : results.delta[i]) << "," << results.nearestHigher[i] << ","
                << results.isSeed[i] << "," << results.clusterIndex[i] << "\n";
      }

      clueOut.close();

      std::cout << "Ouput was saved in " << outDir << std::endl;
    }

    ctx.emplace(event, resultsToken_, std::move(results));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEOutputProducer);
