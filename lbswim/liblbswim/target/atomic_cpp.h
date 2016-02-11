// -*- mode: C++; -*-
#ifndef TARGET_ATOMIC_CPP_H
#define TARGET_ATOMIC_CPP_H

#include "./atomic.h"

namespace target {
  namespace atomic {
    template <typename T>
    __target__ T increment(T& value, T increment) {
      T old = value;
      // without OpenMP the pragma is just ignored
#pragma omp atomic
      value += increment;
      return old;
    }
  }
}

#endif
