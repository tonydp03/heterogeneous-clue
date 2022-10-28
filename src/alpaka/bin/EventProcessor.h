#ifndef EventProcessor_h
#define EventProcessor_h

#include <filesystem>
#include <string>
#include <vector>

#include "AlpakaCore/backend.h"
#include "Framework/EventSetup.h"

#include "PluginManager.h"
#include "StreamSchedule.h"
#include "Source.h"

namespace edm {
  struct Alternative {
    Alternative() = default;
    Alternative(Backend backend, float weight, std::vector<std::string> path)
        : backend{backend}, weight{weight}, path{std::move(path)} {}

    Backend backend;
    float weight;
    std::vector<std::string> path;
  };

  using Alternatives = std::vector<Alternative>;

  class EventProcessor {
  public:
    explicit EventProcessor(int dims,
                            int maxEvents,
                            int runForMinutes,
                            int numberOfStreams,
                            Alternatives alternatives,
                            std::vector<std::string> const& esproducers,
                            std::filesystem::path const& inputFile,
                            std::filesystem::path const& configFile,
                            bool validation);

    int maxEvents() const { return source_->maxEvents(); }
    int processedEvents() const { return source_->processedEvents(); }
    std::vector<std::pair<Backend, int>> const& backends() const { return streamsPerBackend_; }

    void runToCompletion();

    void endJob();

  private:
    edmplugin::PluginManager pluginManager_;
    ProductRegistry registry_;
    Source* source_;
    EventSetup eventSetup_;
    std::vector<StreamSchedule> schedules_;
    std::vector<std::pair<Backend, int>> streamsPerBackend_;
  };
}  // namespace edm

#endif  // EventProcessor_h
