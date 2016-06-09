// -*- mode: C++; -*-
#ifndef TARGET_TARGETPP_H
#define TARGET_TARGETPP_H

#include "./func_attr.h"
#include "../array.h"
#include "target.h"
#include <utility>


#ifdef __INTEL_COMPILER
#include "./vector_intel.h"

#else
#ifdef __GNUC__
#include "./vector_gnu.h"

#else
#error "Can't ID this compiler"
#endif

#endif

// Backend-independent things
namespace target {
  // typesafe malloc
  template<typename T>
  void malloc(T*& ptr, const size_t n = 1);
  // add free
  template<typename T>
  void free(T* ptr);
  
  template<typename T>
  void copyIn(T* targetData, const T* data, const size_t n = 1);
  template<typename T>
  void copyOut(T* data, const T* targetData, const size_t n = 1);
  
  inline void synchronize() {
    targetSynchronize();
  }

}

#ifdef TARGET_MODE_CUDA
#define TARGET_BACKEND_CUDA
#endif
#if defined(TARGET_MODE_OPENMP) || defined(TARGET_MODE_VANILLA)
#define TARGET_BACKEND_CPP
#endif


#if defined(TARGET_MODE_CUDA)
// CUDA backend
#include "./cuda_backend.hpp"
#define TARGET_DEVICE_CODE __CUDACC__
#define TARGET_TLP_PRAGMA
#define TARGET_ILP_PRAGMA

#elif defined(TARGET_BACKEND_CPP)
// C++ backend
#include "./cpp_backend.hpp"

#if defined(TARGET_MODE_OPENMP)
#define TARGET_TLP_PRAGMA   _Pragma("omp parallel for")
#define TARGET_ILP_PRAGMA _Pragma("omp simd")

#else
#define TARGET_TLP_PRAGMA
#define TARGET_ILP_PRAGMA TARGET_SIMD_PRAGMA
#endif

#else
#error "TARGET_MODE not defined!"

#endif

#include "./targetpp.hpp"

#define TARGET_KERNEL_DECLARE(name, nd, vl, ...)			\
  struct name : public ::target::Kernel<name, nd, vl> {	\
    using ::target::Kernel<name, nd, vl>::Kernel;	\
    __target__ void Run(__VA_ARGS__);					\
  }
#define TARGET_KERNEL_DEFINE(name, ...)		\
  __target__ void name::Run(__VA_ARGS__)
  
#define FOR_TLP(threadSpace)						\
  const auto threadSpace##_TLP_end = indexSpace->end();			\
  TARGET_TLP_PRAGMA							\
  for (auto threadSpace = indexSpace->begin(); threadSpace < threadSpace##_TLP_end; ++threadSpace)

#define FOR_ILP(index)					\
  TARGET_ILP_PRAGMA					\
  for (size_t index = 0; index < VecLen(); ++index)

#endif
