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

class CLUEOutputProducer : public edm::EDProducer {
public:
  explicit CLUEOutputProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  edm::EDGetTokenT<PointsCloudSerial> clustersToken_;
};

CLUEOutputProducer::CLUEOutputProducer(edm::ProductRegistry& reg) : clustersToken_(reg.consumes<PointsCloudSerial>()) {}

void CLUEOutputProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto outDir = eventSetup.get<std::filesystem::path>();
  auto const& results = event.get(clustersToken_);

  Parameters par;
  par = eventSetup.get<Parameters>();
  if (par.produceOutput) {
    auto const& outDir = eventSetup.get<std::filesystem::path>();
    std::string output_file_name = create_outputfileName(event.eventID(), par.dc, par.rhoc, par.outlierDeltaFactor);
    std::filesystem::path outFile = outDir / output_file_name;

    std::ofstream clueOut(outFile);

    clueOut << "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
    for (unsigned int i = 0; i < results.x.size(); i++) {
      clueOut << i << "," << results.x[i] << "," << results.y[i] << "," << results.layer[i] << "," << results.weight[i]
              << "," << results.rho[i] << "," << (results.delta[i] > 999 ? 999 : results.delta[i]) << ","
              << results.nearestHigher[i] << "," << results.isSeed[i] << "," << results.clusterIndex[i] << "\n";
    }

    clueOut.close();

    std::cout << "Ouput was saved in " << outFile << std::endl;
  }
}

DEFINE_FWK_MODULE(CLUEOutputProducer);
