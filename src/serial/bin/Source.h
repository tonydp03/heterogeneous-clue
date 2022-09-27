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
#include "DataFormats/ClusterCollection.h"
#include "DataFormats/LayerTilesConstants.h"

namespace edm {

  class Source {
  public:
    explicit Source(int maxEvents,
                    int runForMinutes,
                    ProductRegistry& reg,
                    std::filesystem::path const& inputFile,
                    bool validation);

    virtual ~Source() = default;
    void startProcessing();

    int maxEvents() const { return maxEvents_; }
    int processedEvents() const { return numEvents_; }

    // thread safe
    virtual std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg) = 0;

  protected:
    int maxEvents_;

    // these are all for the mode where the processing length is limited by time
    int const runForMinutes_;
    std::chrono::steady_clock::time_point startTime_;
    std::mutex timeMutex_;
    std::atomic<int> numEventsTimeLastCheck_ = 0;
    std::atomic<bool> shouldStop_ = false;

    std::atomic<int> numEvents_ = 0;
    bool validation_;
  };

  class Source2D : public Source {
  public:
    explicit Source2D(int maxEvents,
                      int runForMinutes,
                      ProductRegistry& reg,
                      std::filesystem::path const& inputFile,
                      bool validation);
    //thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg) override;

  private:
    EDPutTokenT<PointsCloud> const cloudToken_;
    std::vector<PointsCloud> cloud_;
  };

  class Source3D : public Source {
  public:
    explicit Source3D(int maxEvents,
                      int runForMinutes,
                      ProductRegistry& reg,
                      std::filesystem::path const& inputFile,
                      bool validation);
    //thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg) override;

  private:
    EDPutTokenT<ClusterCollection> const clusterToken_;
    std::vector<ClusterCollection> clusters_;
  };
}  // namespace edm

#endif