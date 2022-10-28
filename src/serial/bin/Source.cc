#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "Source.h"

struct Point {
  float x;
  float y;
  float layer;
  float weight;
};

struct PointClus {
  float x;
  float y;
  float z;
  float eta;
  float phi;
  float r_over_absz;
  float radius;
  float layer;
  float energy;
  float isSilicon;
};

namespace {

  PointsCloud readRaw2D(std::ifstream &inputFile, uint32_t n_points) {
    PointsCloud data;
    data.n = n_points;
    Point raw;
    for (unsigned int ipoint = 0; ipoint < n_points; ++ipoint) {
      inputFile.read(reinterpret_cast<char *>(&raw), sizeof(Point));
      data.x.emplace_back(raw.x);
      data.y.emplace_back(raw.y);
      data.layer.emplace_back(raw.layer);
      data.weight.emplace_back(raw.weight);
    }
    return data;
  }

  ClusterCollection readRaw3D(std::ifstream &inputFile, uint32_t n_points) {
    ClusterCollection data;
    PointClus raw;
    for (unsigned int ipoint = 0; ipoint < n_points; ++ipoint) {
      inputFile.read(reinterpret_cast<char *>(&raw), sizeof(PointClus));
      data.x.emplace_back(raw.x);
      data.y.emplace_back(raw.y);
      data.z.emplace_back(raw.z);
      data.eta.emplace_back(raw.eta);
      data.phi.emplace_back(raw.phi);
      data.r_over_absz.emplace_back(raw.r_over_absz);
      data.radius.emplace_back(raw.radius);
      data.layer.emplace_back(raw.layer);
      data.energy.emplace_back(raw.energy);
      data.isSilicon.emplace_back(raw.isSilicon);
    }
    return data;
  }

  PointsCloud readToyDetectors(std::filesystem::path const &toyDetector) {
    PointsCloud data;
    for (int l = 0; l < NLAYERS; l++) {
      std::ifstream in(toyDetector, std::ios::binary);
      while (true) {
        Point raw;
        in.read((char *)&raw, sizeof(Point));
        if (in.eof()) {
          break;
        }
        data.x.emplace_back(raw.x);
        data.y.emplace_back(raw.y);
        data.layer.emplace_back(raw.layer + l);
        data.weight.emplace_back(raw.weight);
      }
      in.close();
    }
    data.n = data.x.size();
    return data;
  }
}  // namespace

namespace edm {
  Source::Source(
      int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &inputFile, bool validation)
      : maxEvents_(maxEvents), runForMinutes_(runForMinutes), validation_(validation) {}

  Source2D::Source2D(
      int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &inputFile, bool validation)
      : Source(maxEvents, runForMinutes, reg, inputFile, validation), cloudToken_(reg.produces<PointsCloud>()) {
    std::string input(inputFile);
    if (input.find("toyDetector") != std::string::npos) {
      cloud_.emplace_back(readToyDetectors(inputFile));
      if (runForMinutes_ < 0 and maxEvents_ < 0) {
        maxEvents_ = 10;
      }
    } else {
      std::ifstream in_raw(inputFile, std::ios::binary);
      uint32_t n_points;
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&n_points), sizeof(uint32_t));

      while (not in_raw.eof()) {
        in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
        cloud_.emplace_back(readRaw2D(in_raw, n_points));

        // next event
        in_raw.exceptions(std::ifstream::badbit);
        in_raw.read(reinterpret_cast<char *>(&n_points), sizeof(uint32_t));
      }
      if (runForMinutes_ < 0 and maxEvents_ < 0) {
        maxEvents_ = cloud_.size();
      }
    }
    if (validation_) {
      for (unsigned int i = 0; i != cloud_.size(); ++i) {
        assert(cloud_[i].n == cloud_[i].x.size());
        assert(cloud_[i].x.size() == cloud_[i].y.size());
        assert(cloud_[i].y.size() == cloud_[i].layer.size());
        assert(cloud_[i].layer.size() == cloud_[i].weight.size());
      }
    }
  }

  Source3D::Source3D(
      int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &inputFile, bool validation)
      : Source(maxEvents, runForMinutes, reg, inputFile, validation), clusterToken_(reg.produces<ClusterCollection>()) {
    std::string input(inputFile);
    std::ifstream in_raw(inputFile, std::ios::binary);
    uint32_t n_points;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&n_points), sizeof(uint32_t));

    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      clusters_.emplace_back(readRaw3D(in_raw, n_points));

      // next event
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&n_points), sizeof(uint32_t));
    }
    if (runForMinutes_ < 0 and maxEvents_ < 0) {
      maxEvents_ = clusters_.size();
    }
  }

  void Source::startProcessing() {
    if (runForMinutes_ >= 0) {
      startTime_ = std::chrono::steady_clock::now();
    }
  }

  std::unique_ptr<Event> Source2D::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return nullptr;
    }

    const int old = numEvents_.fetch_add(1);
    const int iev = old + 1;
    if (runForMinutes_ < 0) {
      if (old >= maxEvents_) {
        shouldStop_ = true;
        --numEvents_;
        return nullptr;
      }
    } else {
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(cloud_.size())) {
        std::scoped_lock lock(timeMutex_);
        // if some other thread beat us, no need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(cloud_.size())) {
          auto processingTime = std::chrono::steady_clock::now() - startTime_;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / cloud_.size()) * cloud_.size();
        }
        if (shouldStop_) {
          --numEvents_;
          return nullptr;
        }
      }
    }
    auto ev = std::make_unique<Event>(streamId, iev, reg);
    const int index = old % cloud_.size();

    ev->emplace(cloudToken_, cloud_[index]);

    return ev;
  }

  std::unique_ptr<Event> Source3D::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return nullptr;
    }

    const int old = numEvents_.fetch_add(1);
    const int iev = old + 1;
    if (runForMinutes_ < 0) {
      if (old >= maxEvents_) {
        shouldStop_ = true;
        --numEvents_;
        return nullptr;
      }
    } else {
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(clusters_.size())) {
        std::scoped_lock lock(timeMutex_);
        // if some other thread beat us, no need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(clusters_.size())) {
          auto processingTime = std::chrono::steady_clock::now() - startTime_;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / clusters_.size()) * clusters_.size();
        }
        if (shouldStop_) {
          --numEvents_;
          return nullptr;
        }
      }
    }
    auto ev = std::make_unique<Event>(streamId, iev, reg);
    const int index = old % clusters_.size();

    ev->emplace(clusterToken_, clusters_[index]);

    return ev;
  }
}  // namespace edm
