// -*- mode: C++; -*-
#ifndef TARGET_RAND_CUDA_H
#define TARGET_RAND_CUDA_H

#ifdef __NVCC__
#include <curand_kernel.h>
#else
struct curandStateXORWOW;
#endif

namespace target {
  namespace rand {
    typedef struct curandStateXORWOW State;
  }
}

#endif
