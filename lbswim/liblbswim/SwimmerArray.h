// -*- mode: C++; -*-
#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "Array.h"
#include "SharedItem.h"
#include "SharedArray.h"

#include "Lists.h"

#include <curand_kernel.h>

class Lattice;

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
  static const size_t BlockSize = 512;
  SwimmerArray(const size_t n, const CommonParams* p);
  void AddForces(Lattice* lat) const;
  void Move(Lattice* lat);
  size_t num;
  SharedItem<CommonParams> common;
  SharedItem<VectorList> r;
  SharedItem<VectorList> v;
  SharedItem<VectorList> n;
  SharedItem<RandList> prng;
};


#endif
