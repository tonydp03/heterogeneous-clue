#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include <cuda_runtime.h>

#include "DataFormats/CLUE_config.h"
#include "CUDACore/getCachingDeviceAllocator.h"
#include "EventProcessor.h"
#include "PosixClockGettime.h"

namespace {
  void print_help(std::string const& name) {
    std::cout << name
              << ": [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--inputFile "
                 "PATH] [--configFile] [--transfer] [--validation] "
                 "[--empty]\n\n"
              << "Options\n"
              << " --numberOfThreads   Number of threads to use (default 1, use 0 to use all CPU cores)\n"
              << " --numberOfStreams   Number of concurrent events (default 0 = numberOfThreads)\n"
              << " --maxEvents         Number of events to process (default -1 for all events in the input file)\n"
              << " --inputFile         Path to the input file to cluster with CLUE (default is set to "
                 "data/input/raw2D.bin)'\n"
              << " --configFile        Path to the config file with the parameters (dc, rhoc, outlierDeltaFactor, "
                 "produceOutput) to run CLUE (implies --transfer, default 'config/hgcal_config.csv' in the directory "
                 "of the executable)\n"
              << " --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)\n"
              << " --validation        Run (rudimentary) validation at the end (implies --transfer)\n"
              << " --empty             Ignore all producers (for testing only)\n"
              << std::endl;
  }
}  // namespace

int main(int argc, char** argv) {
  // Parse command line arguments
  std::vector<std::string> args(argv, argv + argc);
  int numberOfThreads = 1;
  int numberOfStreams = 0;
  int maxEvents = -1;
  int runForMinutes = -1;
  std::filesystem::path inputFile;
  std::filesystem::path configFile;
  bool transfer = false;
  bool validation = false;
  bool empty = false;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
    } else if (*i == "--numberOfThreads") {
      ++i;
      numberOfThreads = std::stoi(*i);
    } else if (*i == "--numberOfStreams") {
      ++i;
      numberOfStreams = std::stoi(*i);
    } else if (*i == "--maxEvents") {
      ++i;
      maxEvents = std::stoi(*i);
    } else if (*i == "--runForMinutes") {
      ++i;
      runForMinutes = std::stoi(*i);
    } else if (*i == "--inputFile") {
      ++i;
      inputFile = *i;
    } else if (*i == "--configFile") {
      ++i;
      configFile = *i;
    } else if (*i == "--transfer") {
      transfer = true;
    } else if (*i == "--validation") {
      transfer = true;
      validation = true;
      std::string fileName(inputFile);
      if (fileName.find("toyDetector") != std::string::npos) {
        configFile = std::filesystem::path(args[0]).parent_path() / "config" / "toyDetector_config.csv";
      }
    } else if (*i == "--empty") {
      empty = true;
    } else {
      std::cout << "Invalid parameter " << *i << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    }
  }
  if (maxEvents >= 0 and runForMinutes >= 0) {
    std::cout << "Got both --maxEvents and --runForMinutes, please give only one of them" << std::endl;
    return EXIT_FAILURE;
  }
  if (numberOfThreads == 0) {
    numberOfThreads = tbb::info::default_concurrency();
  }
  if (numberOfStreams == 0) {
    numberOfStreams = numberOfThreads;
  }
  if (inputFile.empty()) {
    inputFile = std::filesystem::path(args[0]).parent_path() / "data/input/raw2D.bin";
  }
  if (not std::filesystem::exists(inputFile)) {
    std::cout << "Input file '" << inputFile << "' does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  if (configFile.empty()) {
    configFile = std::filesystem::path(args[0]).parent_path() / "config" / "hgcal_config.csv";
  }
  if (not std::filesystem::exists(configFile)) {
    std::cout << "Config file '" << configFile << "' does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  int numberOfDevices;
  auto status = cudaGetDeviceCount(&numberOfDevices);
  if (cudaSuccess != status) {
    std::cout << "Failed to initialize the CUDA runtime";
    return EXIT_FAILURE;
  }
  std::cout << "Found " << numberOfDevices << " devices" << std::endl;

#if CUDA_VERSION >= 11020
  // Initialize the CUDA memory pool
  uint64_t threshold = cms::cuda::allocator::minCachedBytes();
  for (int device = 0; device < numberOfDevices; ++device) {
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, device);
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  }
#endif

  Parameters par;
  std::ifstream iFile(configFile);
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

  std::cerr << "Running CLUE algorithm with the following parameters: \n";
  std::cerr << "dc = " << par.dc << '\n';
  std::cerr << "rhoc = " << par.rhoc << '\n';
  std::cerr << "outlierDeltaFactor = " << par.outlierDeltaFactor << std::endl;

  if (par.produceOutput) {
    transfer = true;
    std::cerr << "Producing output at the end" << std::endl;
  }
  // Initialize EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  if (not empty) {
    edmodules = {"CLUECUDAClusterizer"};
    esmodules = {"CLUECUDAClusterizerESProducer"};
    if (transfer) {
      // add modules for transfer
      edmodules.emplace_back("CLUEOutputProducer");
      esmodules.emplace_back("CLUEOutputESProducer");
    }
    if (validation) {
      esmodules.emplace_back("CLUEValidatorESProducer");
      edmodules.emplace_back("CLUEValidator");
    }
  }
  edm::EventProcessor processor(maxEvents,
                                runForMinutes,
                                numberOfStreams,
                                std::move(edmodules),
                                std::move(esmodules),
                                inputFile,
                                configFile,
                                validation);
  maxEvents = processor.maxEvents();

  if (runForMinutes < 0) {
    std::cout << "Processing " << processor.maxEvents() << " events,";
  } else {
    std::cout << "Processing for about " << runForMinutes << " minutes,";
  }
  {
    std::cout << " with " << numberOfStreams << " concurrent events (";
    std::cout << ") and " << numberOfThreads << " threads." << std::endl;
  }

  // Initialize the TBB thread pool
  tbb::global_control tbb_max_threads{tbb::global_control::max_allowed_parallelism,
                                      static_cast<std::size_t>(numberOfThreads)};

  // Run work
  auto cpu_start = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto start = std::chrono::high_resolution_clock::now();
  try {
    tbb::task_arena arena(numberOfThreads);
    arena.execute([&] { processor.runToCompletion(); });
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }
  auto cpu_stop = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Run endJob
  try {
    processor.endJob();
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }

  // Work done, report timing
  auto diff = stop - start;
  auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(diff).count()) / 1e6;
  auto cpu_diff = cpu_stop - cpu_start;
  auto cpu = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(cpu_diff).count()) / 1e6;
  maxEvents = processor.processedEvents();
  std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
            << std::defaultfloat << (maxEvents / time) << " events/s, CPU usage per thread: " << std::fixed
            << std::setprecision(1) << (cpu / time / numberOfThreads * 100) << "%" << std::endl;
  return EXIT_SUCCESS;
}
