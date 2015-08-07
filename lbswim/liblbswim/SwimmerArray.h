#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "Lattice.h"

struct SwimmerArray {
  SwimmerArray(int n, double hydro);
  void AddForces(Lattice* lat);
  void Move(Lattice* lat);
  int num;
  double hydroRadius;
  SharedArray<double> r;
  SharedArray<double> v;
  SharedArray<double> n;
  SharedArray<double> P;
  SharedArray<double> a;
  SharedArray<double> l;
};


#endif
