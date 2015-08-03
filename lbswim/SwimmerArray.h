#ifndef SWIMMER_ARRAY_H
#define SWIMMER_ARRAY_H

#include "Lattice.h"

#ifdef __cplusplus 
extern "C" {
#endif

struct SwimmerArrayImpl {
  int num;
  double hydroRadius;
  
  double* r;
  double* v;
  double* n;
  double* P;
  double* a;
  double* l;
};

struct SwimmerArray {
  SwimmerArrayImpl* h;
  SwimmerArrayImpl* d;
  SwimmerArrayImpl* d_h;
};

SwimmerArray* SwimmerArrayNew(int n, double hydro);
void SwimmerArrayDel(SwimmerArray* sa);

void SwimmerArrayAddForces(SwimmerArray* sa, Lattice* lat);
void SwimmerArrayMove(SwimmerArray* sa, Lattice* lat);

#ifdef __cplusplus 
}
#endif

#endif
