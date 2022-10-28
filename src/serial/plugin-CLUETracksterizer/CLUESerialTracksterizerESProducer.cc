#include <fstream>
#include <memory>
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"
#include "DataFormats/CLUE_config.h"

class CLUESerialTracksterizerESProducer : public edm::ESProducer {
public:
  CLUESerialTracksterizerESProducer(std::filesystem::path const& config_file) : data_{config_file} {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void CLUESerialTracksterizerESProducer::produce(edm::EventSetup& eventSetup) {
  Parameters par;
  // need another file for parameters
  std::ifstream iFile(data_);
  std::string value = "";
  while (getline(iFile, value, ',')) {
    par.dc = std::stof(value);
    getline(iFile, value, ',');
    par.rhoc = std::stof(value);
    getline(iFile, value, ',');
    par.outlierDeltaFactor = std::stof(value);
    getline(iFile, value);
    par.produceOutput = static_cast<bool>(std::stoi(value));
  }
  iFile.close();

  auto parameters = std::make_unique<Parameters>(par);
  eventSetup.put(std::move(parameters));
}

DEFINE_FWK_EVENTSETUP_MODULE(CLUESerialTracksterizerESProducer);