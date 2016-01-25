// -*- mode: C++; -*-
#ifndef LATTICEADDRESSING_H
#define LATTICEADDRESSING_H

struct LatticeAddressing {
  LatticeAddressing(int nx, int ny, int nz);
  int size[DQ_d];
  int strides[DQ_d];
  int n;
};

#endif
