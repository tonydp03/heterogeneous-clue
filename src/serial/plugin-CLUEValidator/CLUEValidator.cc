#include <iostream>
#include <fstream>
#include <type_traits>
#include <unordered_map>
#include <string>

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "CLUEValidatorTypes.h"

std::vector<float> CLAMPED(std::vector<float> in, float upperLimit) {
  std::vector<float> out(in);
  for (size_t i = 0; i < out.size(); i++)
    if (out[i] > upperLimit)
      out[i] = upperLimit;
  return out;
}

class CLUEOutputProducer;
class CLUEValidator : public edm::EDProducer {
public:
  explicit CLUEValidator(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  template <class T>
  bool arraysAreEqual(std::vector<T>, std::vector<T> trueDataArr);
  bool arraysClustersEqual(const PointsCloudSerial& devicePC, const PointsCloudSerial& truePC);
  std::string checkValidation(std::string const& inputFile);
  void validateOutput(const PointsCloudSerial& pc, std::string trueOutFilePath, Parameters const& par);
  edm::EDGetTokenT<PointsCloudSerial> resultsTokenPC_;
};

CLUEValidator::CLUEValidator(edm::ProductRegistry& reg) : resultsTokenPC_(reg.consumes<PointsCloudSerial>()) {}

template <class T>
bool CLUEValidator::arraysAreEqual(std::vector<T> devicePtr, std::vector<T> trueDataArr) {
  bool sameValue = true;
  for (size_t i = 0; i < devicePtr.size(); i++) {
    if (std::is_same<T, int>::value) {
      sameValue = devicePtr[i] == trueDataArr[i];
    } else {
      const float TOLERANCE = 0.001;
      sameValue = std::abs(devicePtr[i] - trueDataArr[i]) <= TOLERANCE;
    }

    if (!sameValue) {
      std::cout << "failed comparison for i=" << i << ", " << devicePtr[i] << " /= " << trueDataArr[i] << std::endl;
      break;
    }
  }
  return sameValue;
}

bool CLUEValidator::arraysClustersEqual(const PointsCloudSerial& devicePC, const PointsCloudSerial& truePC) {
  std::unordered_map<int, int> clusterIdMap;

  int n = (int)devicePC.x.size();

  for (int i = 0; i < n; i++) {
    if (devicePC.isSeed[i]) {
      clusterIdMap[devicePC.clusterIndex[i]] = truePC.clusterIndex[i];
    }
  }

  bool sameValue = true;
  for (int i = 0; i < n; i++) {
    int originalClusterId = devicePC.clusterIndex[i];
    int mappedClusterId = clusterIdMap[originalClusterId];
    if (originalClusterId == -1)
      mappedClusterId = -1;

    sameValue = (mappedClusterId == truePC.clusterIndex[i]);

    if (!sameValue) {
      std::cout << "failed comparison for i=" << i << ", original=" << originalClusterId
                << ", mapped= " << mappedClusterId << " /= " << truePC.clusterIndex[i] << std::endl;
      break;
    }
  }

  return sameValue;
}

std::string CLUEValidator::checkValidation(std::string const& inputFile) {
  if (inputFile.find("toyDetector_1k") != std::string::npos) {
    return "ref_1k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_2k") != std::string::npos) {
    return "ref_2k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_3k") != std::string::npos) {
    return "ref_3k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_4k") != std::string::npos) {
    return "ref_4k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_5k") != std::string::npos) {
    return "ref_5k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_6k") != std::string::npos) {
    return "ref_6k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_7k") != std::string::npos) {
    return "ref_7k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_8k") != std::string::npos) {
    return "ref_8k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_9k") != std::string::npos) {
    return "ref_9k_20_25_2.csv";
  } else if (inputFile.find("toyDetector_10k") != std::string::npos) {
    return "ref_10k_20_25_2.csv";
  } else {
    return std::string();
  }
}

void CLUEValidator::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto outDataDir = std::make_unique<OutputDirPath>();
  *outDataDir = eventSetup.get<OutputDirPath>();

  auto const& pc = event.get(resultsTokenPC_);
  auto const& par = eventSetup.get<Parameters>();

  if (checkValidation(outDataDir->outFile) != std::string()) {
    auto ref_file = checkValidation(outDataDir->outFile);
    std::filesystem::path ref_path = outDataDir->outFile.parent_path() / "reference";
    std::cout << "Validating output results from " << ref_path / ref_file << std::endl;
    validateOutput(pc, ref_path / ref_file, par);
  } else {
    std::cout << "\nThere is no reference output data for the input file selected.\n";
    std::cout << "Please select one of the toyDetectors input files to validate the plugin results\n" << std::endl;
  }
}

void CLUEValidator::validateOutput(const PointsCloudSerial& pc, std::string trueOutFilePath, Parameters const& par) {
  PointsCloudSerial truePC;
  std::ifstream iTrueDataFile(trueOutFilePath);
  std::string value = "";
  // Get Header Line
  getline(iTrueDataFile, value);
  int n = 1;
  while (getline(iTrueDataFile, value, ',')) {
    try {
      getline(iTrueDataFile, value, ',');
      truePC.x.push_back(std::stof(value));
      getline(iTrueDataFile, value, ',');
      truePC.y.push_back(std::stof(value));
      getline(iTrueDataFile, value, ',');
      truePC.layer.push_back(std::stoi(value));
      getline(iTrueDataFile, value, ',');
      truePC.weight.push_back(std::stof(value));
      getline(iTrueDataFile, value, ',');
      truePC.rho.push_back(std::stof(value));
      getline(iTrueDataFile, value, ',');
      truePC.delta.push_back(std::stof(value));
      getline(iTrueDataFile, value, ',');
      truePC.nearestHigher.push_back(std::stoi(value));
      getline(iTrueDataFile, value, ',');
      truePC.isSeed.push_back(std::stoi(value));
      getline(iTrueDataFile, value);
      truePC.clusterIndex.push_back(std::stoi(value));
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      std::cout << "Bad Input: '" << value << "' in line " << n << std::endl;
      break;
    }
    n++;
  }
  iTrueDataFile.close();

  // input variables
  assert(arraysAreEqual(pc.x, truePC.x));
  std::cout << "Output x -> Ok" << '\n';
  assert(arraysAreEqual(pc.y, truePC.y));
  std::cout << "Output y -> Ok" << '\n';
  assert(arraysAreEqual(pc.layer, truePC.layer));
  std::cout << "Output layer -> Ok" << '\n';
  assert(arraysAreEqual(pc.weight, truePC.weight));
  std::cout << "Output weight -> Ok" << '\n';
  std::cout << "Input variables are correct!" << std::endl;

  if (par.dc == 20 && par.rhoc == 25 && par.outlierDeltaFactor == 2) {
    // result variables
    std::cout << "Using the same parameters as reference file, checking output results" << '\n';
    assert(arraysAreEqual(pc.rho, truePC.rho));
    std::cout << "Output rho -> Ok" << '\n';
    assert(arraysAreEqual(CLAMPED(pc.delta, 999), truePC.delta));
    std::cout << "Output delta -> Ok" << '\n';
    assert(arraysAreEqual(pc.nearestHigher, truePC.nearestHigher));
    std::cout << "Output nearestHigher -> Ok" << '\n';
    assert(arraysAreEqual(pc.isSeed, truePC.isSeed));
    std::cout << "Output isSeed -> Ok" << '\n';
    assert(arraysClustersEqual(pc, truePC));
    std::cout << "Output cluster IDs -> Ok" << '\n';
    std::cout << "CLUE output is correct!" << std::endl;
  } else {
    std::cout << "Parameters different from the ones used in the reference file, cannot check results" << std::endl;
  }
}

DEFINE_FWK_MODULE(CLUEValidator);
