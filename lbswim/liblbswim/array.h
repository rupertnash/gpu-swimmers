// -*- mode: C++; -*-
#ifndef array_H
#define array_H
#include <cstdlib>
#include "target/func_attr.h"

/* Minimal CUDA-compatible std::array replacement */

template <typename T, size_t N>
class array
{
  T data[N];
  __targetBoth__ void Init(const size_t& i, const T& a) {
    data[i] = a;
  }
  template<typename... ARGS>
  __targetBoth__ void Init(const size_t& i, const T& a, ARGS... args) {
    data[i] = a;
    Init(i+1, args...);
  }

public:
  __targetBoth__ static constexpr size_t size() {
    return N;
  }
  
  __targetBoth__ array() {
    for (size_t i = 0; i<N; ++i)
      data[i] = T();
  }
  
  template<typename... ARGS>
  __targetBoth__ array(ARGS... args) {
    static_assert(sizeof...(args) == N, "Wrong number of arguments");
    Init(0, args...);
  }
    
  __targetBoth__ T& operator[](size_t i) {
    return data[i];
  }
  __targetBoth__ const T& operator[](size_t i) const {
    return data[i];
  }
};

template <typename T, size_t N>
__targetBoth__ bool operator==(const array<T,N>& a, const array<T,N>& b) {

  for (size_t i = 0; i< N; ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

template <typename T, size_t N>
__targetBoth__ bool operator!=(const array<T,N>& a, const array<T,N>& b) {
  return !(a == b);
}
#endif
