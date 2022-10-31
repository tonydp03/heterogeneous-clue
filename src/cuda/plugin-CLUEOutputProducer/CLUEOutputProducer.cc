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

#include "CUDACore/Product.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/ScopedContext.h"

#include "CUDADataFormats/PointsCloudCUDA.h"

class CLUEOutputProducer : public edm::EDProducer {
public:
  explicit CLUEOutputProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  edm::EDGetTokenT<cms::cuda::Product<PointsCloudCUDA>> deviceClustersToken_;
  edm::EDPutTokenT<cms::cuda::Product<PointsCloud>> resultsToken_;
  edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
};

CLUEOutputProducer::CLUEOutputProducer(edm::ProductRegistry& reg)
    : deviceClustersToken_(reg.consumes<cms::cuda::Product<PointsCloudCUDA>>()),
      resultsToken_(reg.produces<cms::cuda::Product<PointsCloud>>()),
      pointsCloudToken_(reg.consumes<PointsCloud>()) {}

void CLUEOutputProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto outDir = eventSetup.get<std::filesystem::path>();
  auto results = event.get(pointsCloudToken_);
  auto const& pcProduct = event.get(deviceClustersToken_);
  cms::cuda::ScopedContextProduce ctx{pcProduct};
  auto const& device_clusters = ctx.get(pcProduct);
  auto stream = ctx.stream();

  results.outResize();
  cudaCheck(cudaMemcpyAsync(
      results.rho.data(), device_clusters.rho.get(), sizeof(float) * results.x.size(), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(results.delta.data(),
                            device_clusters.delta.get(),
                            sizeof(float) * results.x.size(),
                            cudaMemcpyDeviceToHost,
                            stream));
  cudaCheck(cudaMemcpyAsync(results.nearestHigher.data(),
                            device_clusters.nearestHigher.get(),
                            sizeof(float) * results.x.size(),
                            cudaMemcpyDeviceToHost,
                            stream));
  cudaCheck(cudaMemcpyAsync(results.isSeed.data(),
                            device_clusters.isSeed.get(),
                            sizeof(int) * results.x.size(),
                            cudaMemcpyDeviceToHost,
                            stream));
  cudaCheck(cudaMemcpyAsync(results.clusterIndex.data(),
                            device_clusters.clusterIndex.get(),
                            sizeof(int) * results.x.size(),
                            cudaMemcpyDeviceToHost,
                            stream));
  cudaStreamSynchronize(stream);

  std::cout << "Data transferred back to host" << std::endl;

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

  ctx.emplace(event, resultsToken_, std::move(results));
}

DEFINE_FWK_MODULE(CLUEOutputProducer);
