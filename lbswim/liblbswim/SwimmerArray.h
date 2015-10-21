#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "Lattice.h"
#include <curand_kernel.h>
typedef struct curandStateXORWOW RandState;

struct CommonParams {
  double P;
  double l;
  double alpha;
  double mobility;
  bool translational_advection_off;
  bool rotational_advection_off;
  unsigned long long seed;
};

struct SwimmerArray {
  static const int BlockSize = 512;
  SwimmerArray(const int n, const CommonParams* p);
  void AddForces(Lattice* lat) const;
  void Move(Lattice* lat);
  int num;
  SharedItem<CommonParams> common;
  SharedArray<double> r;
  SharedArray<double> v;
  SharedArray<double> n;
  SharedArray<RandState> prng;
};


#endif
