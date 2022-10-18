#ifndef Source_h
#define Source_h

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "Framework/Event.h"
#include "DataFormats/PointsCloud.h"
#include "DataFormats/LayerTilesConstants.h"

namespace edm {
  class Source {
  public:
    explicit Source(int maxEvents,
                    int runForMinutes,
                    ProductRegistry& reg,
                    std::filesystem::path const& inputFile,
                    bool validation);

    void startProcessing();

    int maxEvents() const { return maxEvents_; }
    int processedEvents() const { return numEvents_; }

    // thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg);

  private:
    int maxEvents_;

    // these are all for the mode where the processing length is limited by time
    int const runForMinutes_;
    std::chrono::steady_clock::time_point startTime_;
    std::mutex timeMutex_;
    std::atomic<int> numEventsTimeLastCheck_ = 0;
    std::atomic<bool> shouldStop_ = false;

    std::atomic<int> numEvents_ = 0;
    EDPutTokenT<PointsCloud> const cloudToken_;
    std::vector<PointsCloud> cloud_;
    bool validation_;
  };
}  // namespace edm

#endif