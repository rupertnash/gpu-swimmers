// -*- mode: C++; -*-
#ifndef TRACER_ARRAY_H
#define TRACER_ARRAY_H

#include "Lattice.h"

struct TracerArray {
  static const size_t BlockSize = 512;
  TracerArray(const size_t n);
  void Move(Lattice* lat);
  size_t num;
  SharedItem<VectorList> r;
  SharedItem<VectorList> s;
  SharedItem<VectorList> v;
};


#endif
