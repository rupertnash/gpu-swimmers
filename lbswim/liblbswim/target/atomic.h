// -*- mode: C++; -*-
#ifndef TARGET_ATOMIC_H
#define TARGET_ATOMIC_H

#include "./func_attr.h"

namespace target {
  namespace atomic {
    template <typename T>
    __target__ T increment(T&, T);
  }
}

#if defined(TARGET_MODE_CUDA)
// CUDA 
#include "./atomic_cuda.h"

#elif defined(TARGET_MODE_OPENMP)
// OpenMP C++
#include "./atomic_omp.h"

#elif defined(TARGET_MODE_VANILLA)
// Vanilla C++
#include "./atomic_vanilla.h"

#else
#error "TARGET_MODE not defined!"

#endif
#endif
