// -*- mode: C++; -*-
#ifndef TARGET_ATOMIC_VANILLA_H
#define TARGET_ATOMIC_VANILLA_H

#include "./atomic.h"

namespace target {
  namespace atomic {
    template <typename T>
    __target__ T increment(T& value, T increment) {
      T old = value;
      value += increment;
      return old;
    }
  }
}

#endif
