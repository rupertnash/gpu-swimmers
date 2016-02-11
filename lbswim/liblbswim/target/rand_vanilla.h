// -*- mode: C++; -*-
#ifndef TARGET_RAND_VANILLA_H
#define TARGET_RAND_VANNILA_H

#include <random>

namespace target {
  namespace rand {
    typedef std::mt19937_64 EngineType;

    class State {
      EngineType engine;
      std::uniform_real_distribution<float> uniform;
      std::normal_distribution<double> normal;

      friend void init(unsigned long long seed, unsigned long long sequence, State* state);
      friend float uniform(State* state);
      friend double normal_double(State* state);
    };
  }
}

#endif
