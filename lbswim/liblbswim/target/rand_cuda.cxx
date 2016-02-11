#include <curand_kernel.h>
#include "./rand.h"

namespace target {
  namespace rand {
    __target__ void init(unsigned long long seed, unsigned long long sequence, State* state) {
      curand_init(seed, sequence, 0, state);
    }
    
    __target__ float uniform(State *state) {
      return curand_uniform(state);
    }
    
    __target__ double normal_double(State* state) {
      return curand_normal_double(state);
    }
  }
}
