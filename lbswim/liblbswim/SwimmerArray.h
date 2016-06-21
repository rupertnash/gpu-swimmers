// -*- mode: C++; -*-
#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "target/NdArray.h"
#include "target/SharedItem.h"
#include "target/SharedNdArray.h"

#include "Lists.h"

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
  target::SharedItem<CommonParams> common;
  target::SharedItem<VectorList> r;
  target::SharedItem<VectorList> v;
  target::SharedItem<VectorList> n;
  target::SharedItem<RandList> prng;
};


#endif
