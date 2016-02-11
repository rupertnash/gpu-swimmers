#include "./rand.h"

namespace target {
  namespace rand {
    
    __target__ void init(unsigned long long seed, unsigned long long sequence, State* state) {
      state->engine = EngineType(seed + sequence);
      state->uniform = std::uniform_real_distribution<float>();
      state->normal = std::normal_distribution<double>();
    }
    
    __target__ float uniform(State *state) {
      return state->uniform(state->engine);
    }
    
    __target__ double normal_double(State* state) {
      return state->normal(state->engine);
    }
    
  }
}
