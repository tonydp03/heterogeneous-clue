#include <fstream>
#include <memory>
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"
#include "CLUEValidatorTypes.h"

class CLUEValidatorESProducer : public edm::ESProducer {
public:
  CLUEValidatorESProducer(std::filesystem::path const& inputFile) : data_{inputFile} {
    outFileName_ = getOutputFileName(inputFile);
  }
  void produce(edm::EventSetup& eventSetup);

  std::string getOutputFileName(std::filesystem::path const& inputFile) {
    std::string fileName = inputFile.filename();
    fileName.erase(fileName.end() - 4, fileName.end());
    fileName.append("_output.csv");
    return fileName;
  }

private:
  std::string outFileName_;
  std::filesystem::path data_;
};

void CLUEValidatorESProducer::produce(edm::EventSetup& eventSetup) {
  auto outDir =
      std::make_unique<OutputDirPath>(OutputDirPath{data_.parent_path().parent_path() / "output" / outFileName_});
  eventSetup.put(std::move(outDir));
}

DEFINE_FWK_EVENTSETUP_MODULE(CLUEValidatorESProducer);
