// -*- mode: C++; -*-
#ifndef TARGET_TARGETPP_H
#define TARGET_TARGETPP_H

#include "./func_attr.h"
#include "../array.h"

// Backend-independent things
namespace target {
  // typesafe malloc
  template<typename T>
  void malloc(T*& ptr, const size_t n = 1);
  // add free
  template<typename T>
  void free(T*& ptr);
  
  template<typename T>
  void copyIn(T* targetData, const T* data, const size_t n = 1);
  template<typename T>
  void copyOut(T* data, const T* targetData, const size_t n = 1);

  inline void synchronize() {
    targetSynchronize();
  }
}

#if defined(TARGET_MODE_CUDA)
// CUDA backend
#include "./cuda_backend.hpp"

#elif defined(TARGET_MODE_OPENMP)

// OpenMP C++ backend
#include "./omp_backend.hpp"

#elif defined(TARGET_MODE_OPENMP)
// Vanilla C++ backend
#include "./vanilla_backend.hpp"

#else
#error "TARGET_MODE not defined!"

#endif


#define FOR_TLP2(threadCtx, N) \
  for(auto threadCtx: target::MkContext(N))

#define FOR_TLP(N) FOR_TLP2(__targetThreadCtx, N)

#define FOR_ILP2(threadCtx, i) \
  for(auto i: threadCtx)

#define FOR_ILP(i) FOR_ILP2(__targetThreadCtx, i)


#include "./targetpp.hpp"

#endif
