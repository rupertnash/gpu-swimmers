#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "Lattice.h"

typedef struct curandStateXORWOW RandState;
//struct curandState_t;

struct CommonParams {
  double P;
  double a;
  double l;
  double hydroRadius;
  double alpha;
  unsigned long long seed;
};

struct SwimmerArray {
  SwimmerArray(const int n, const CommonParams* p);
  void AddForces(Lattice* lat) const;
  void Move(Lattice* lat);
  int num;
  SharedItem<CommonParams> common;
  SharedArray<double> r;
  SharedArray<double> v;
  SharedArray<double> n;
  SharedArray<RandState> prng;
  //SharedItem<struct curandState_t> prng;
};


#endif
