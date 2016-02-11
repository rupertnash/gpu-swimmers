// -*- mode: C++; -*-
#ifndef TARGET_RAND_H
#define TARGET_RAND_H

#if defined(TARGET_MODE_CUDA)
// CUDA 
#include "./rand_cuda.h"

#elif defined(TARGET_MODE_OPENMP) || defined(TARGET_MODE_VANILLA)
// OpenMP or plain C++ can use the same implementation
#include "./rand_cpp.h"

#else
#error "TARGET_MODE not defined!"

#endif

#include "./func_attr.h"

namespace target {
  namespace rand {
    __target__ void init(unsigned long long seed, unsigned long long sequence, State* state);
    __target__ float uniform(State *state);
    __target__ double normal_double(State* state);
  }
}

#endif
