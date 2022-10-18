#include <fstream>
#include <filesystem>
#include <memory>
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

class CLUEOutputESProducer : public edm::ESProducer {
public:
  CLUEOutputESProducer(std::filesystem::path const& inputFile) : data_{inputFile} {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void CLUEOutputESProducer::produce(edm::EventSetup& eventSetup) {
  auto outDir = std::make_unique<std::filesystem::path>(data_.parent_path().parent_path() / "output");
  eventSetup.put(std::move(outDir));
}

DEFINE_FWK_EVENTSETUP_MODULE(CLUEOutputESProducer);
