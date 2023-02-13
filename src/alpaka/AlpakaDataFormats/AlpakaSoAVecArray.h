#ifndef AlpakaSoAVecArray_h
#define AlpakaSoAVecArray_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaVecArray.h"
#include "AlpakaVecArrayRef.h"


template<typename T, int nVecArrays, int maxSize>
class AlpakaSoAVecArray {
public:
  using alpakaVect = cms::alpakatools::VecArrayRef<T, maxSize>;

  ALPAKA_FN_HOST_ACC inline constexpr void clear() {
    for (auto& v : vecArrays_)
      v.reset();
  }

  ALPAKA_FN_HOST_ACC inline constexpr void clear(int i) {
    vecArrays_[i].reset();
  }

  ALPAKA_FN_HOST_ACC inline constexpr void setPtrs(int i) {
    vecArrays_[i].setPtr(&sizes_[i]);
  }
  
  ALPAKA_FN_HOST_ACC inline constexpr void init(int i) {
    vecArrays_[i].init();
  }

  ALPAKA_FN_HOST_ACC inline constexpr auto size(int i) {
    return sizes_[i];
  }

  ALPAKA_FN_HOST_ACC inline constexpr alpakaVect& operator[](int id) { return vecArrays_[id]; }

private:
  std::array<alpakaVect, nVecArrays> vecArrays_;
  std::array<int, nVecArrays> sizes_;
};

#endif
