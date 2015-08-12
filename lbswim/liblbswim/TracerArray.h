#ifndef TRACER_ARRAY_H
#define TRACER_ARRAY_H

#include "Lattice.h"

struct TracerArray {
  static const int BlockSize = 512;
  TracerArray(const int n);
  void Move(Lattice* lat);
  int num;
  SharedArray<double> r;
  SharedArray<double> v;
};


#endif
