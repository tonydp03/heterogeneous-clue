#ifndef AlpakaVecArrayRef_h
#define AlpakaVecArrayRef_h

//
// Author: Felice Pantaleo, CERN
//

namespace cms::alpakatools {

  template <class T, int maxSize>
  struct VecArrayRef {
  
    inline constexpr int push_back_unsafe(const T &element) {
      auto previousSize = *p_size;
      (*p_size)++;
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --(*p_size);
        return -1;
      }
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&...args) {
      auto previousSize = *p_size;
      (*p_size)++;
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        --(*p_size);
        return -1;
      }
    }
 
    inline constexpr T &back() const {
      if (*p_size > 0) {
        return m_data[*p_size - 1];
      } else
        return T();  //undefined behaviour
    }

    // thread-safe version of the vector, when used in a  kernel
    template <typename T_Acc>
    ALPAKA_FN_ACC inline constexpr int push_back(const T_Acc &acc, const T &element) {
      auto previousSize = atomicAdd(acc, p_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        atomicSub(acc, p_size, 1, alpaka::hierarchy::Blocks{});
        assert(("Too few elemets reserved"));
        return -1;
      }
    }
    // thread-unsafe version of the vector, when used in a  kernel
    template <typename T_Acc>
      ALPAKA_FN_ACC inline constexpr int push_back_unsafe(const T_Acc &acc, const T &element) {
      auto previousSize = *p_size;
      (*p_size)++;
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --(*p_size);
        return -1;
      }
    }

    template <typename T_Acc, class... Ts>
    ALPAKA_FN_ACC int emplace_back(const T_Acc &acc, Ts &&...args) {
      auto previousSize = atomicAdd(acc, p_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        atomicSub(acc, p_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    template <typename T_Acc, class... Ts>
    ALPAKA_FN_ACC inline T pop_back() {
      if (*p_size > 0) {
        auto previousSize = *p_size--;
        return m_data[previousSize - 1];
      } else
        return T();
    }

    inline constexpr T const *begin() const { return m_data; }
    inline constexpr T const *end() const { return m_data + *p_size; }
    inline constexpr T *begin() { return m_data; }
    inline constexpr T *end() { return m_data + *p_size; }
    inline constexpr int size() const { return *p_size; }
    inline constexpr T &operator[](int i) { return m_data[i]; }
    inline constexpr const T &operator[](int i) const { return m_data[i]; }
    inline constexpr void init() { p_size = &m_size;}
    inline constexpr void reset() { *p_size = 0; }
    inline constexpr int capacity() const { return maxSize; }
    inline constexpr T const *data() const { return m_data; }
    inline constexpr void resize(int size) { *p_size = size; }
    inline constexpr void setPtr(int* ptr) { p_size = ptr; }
    inline constexpr bool empty() const { return 0 == *p_size; }
    inline constexpr bool full() const { return maxSize == *p_size; }
   
    T m_data[maxSize];
   
    int m_size=0;
    int* p_size = nullptr;


  };

}  // end namespace cms::alpakatools

#endif  // AlpakaVecArrayRef_h
