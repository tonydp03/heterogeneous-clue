#ifndef CLUE_config_h
#define CLUE_config_h

#include <vector>
#include <filesystem>
#include <regex>

struct Parameters {
  float dc = 20;
  float rhoc = 25;
  float outlierDeltaFactor = 2;
  bool produceOutput = true;
  // bool putOutputInEvent = false;
};

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

inline std::string create_outputfileName(int const& EventId,
                                         float const& dc,
                                         float const& rhoc,
                                         float const& outlierDeltaFactor) {
  std::string underscore = "_";
  std::string filename = "Event";
  filename.append(underscore);
  filename.append(to_string_with_precision(EventId, 0));
  filename.append(underscore);
  filename.append(to_string_with_precision(dc, 2));
  filename.append(underscore);
  filename.append(to_string_with_precision(rhoc, 2));
  filename.append(underscore);
  filename.append(to_string_with_precision(outlierDeltaFactor, 2));
  filename.append(".csv");
  return filename;
}

#endif