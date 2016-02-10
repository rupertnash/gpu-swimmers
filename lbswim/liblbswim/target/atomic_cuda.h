// -*- mode: C++; -*-
#ifndef TARGET_ATOMIC_CUDA_H
#define TARGET_ATOMIC_CUDA_H

#include "./atomic.h"

namespace target {
  namespace atomic {
    template <typename T>
    __target__ T increment(T& value, T increment) {
      return atomicAdd(&value, increment);
    }

    template<>
    __target__ double increment<double>(double& value, double increment) {
      
      unsigned long long int* address_as_ull = (unsigned long long int*)&value;
      unsigned long long int old = *address_as_ull, assumed;
      do {
	assumed = old;
	old = atomicCAS(address_as_ull,
			assumed,
			__double_as_longlong(increment + __longlong_as_double(assumed)));
	/* Note: uses integer comparison to avoid hang in case of NaN
	 * (since NaN != NaN) */
      } while (assumed != old);
      
      return __longlong_as_double(old);
    }

  }
}

#endif
