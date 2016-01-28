// -*- mode: C++; -*-
#ifndef array_H
#define array_H

#include "cuhelp.h"

#ifdef __CUDACC__

template <typename T, size_t N>
class array
{
  T data[N];
  BOTH void Init(const size_t& i, const T& a) {
    data[i] = a;
  }
  template<typename... ARGS>
  BOTH void Init(const size_t& i, const T& a, ARGS... args) {
    data[i] = a;
    Init(i+1, args...);
  }
public:
  BOTH array() {
    for (size_t i = 0; i<N; ++i)
      data[i] = T();
  }
  
  template<typename... ARGS>
  BOTH array(ARGS... args) {
    static_assert(sizeof...(args) == N, "Wrong number of arguments");
    Init(0, args...);
  }
  
  // BOTH array(std::initializer_list<T> l) {
  //   static_assert(l.size() == N, "Wrong size!");
  //   size_t i = 0
  //   for (; i<N; ++i)
  //     data[i] = l[i];
  // }
  
  BOTH T& operator[](size_t i) {
    return data[i];
  }
  BOTH const T& operator[](size_t i) const {
    return data[i];
  }
};

#else
template <typename T, size_t N>
using array = std::array<T, N>;

#endif

#endif
