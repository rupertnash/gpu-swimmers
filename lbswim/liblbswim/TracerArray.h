// -*- mode: C++; -*-
#ifndef TRACER_ARRAY_H
#define TRACER_ARRAY_H

#include "Lists.h"
#include "target/SharedNdArray.h"

class Lattice;

struct TracerArray {
  static const size_t BlockSize = 512;
  TracerArray(const size_t n);
  void Move(Lattice* lat);
  size_t num;
  target::SharedItem<VectorList> r;
  target::SharedItem<VectorList> s;
  target::SharedItem<VectorList> v;
};


#endif
