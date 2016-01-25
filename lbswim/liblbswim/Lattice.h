// -*- mode: C++; -*-
#ifndef LATTICE_H
#define LATTICE_H

#include "dq.h"
#include "LBParams.h"
#include "SharedItem.h"
#include "SharedArray.h"
#include "LatticeAddressing.h"

struct LDView {
  double* rho;
  double* u;
  double* force;
  double* fOld;
  double* fNew;  
};

struct LatticeData {
  LatticeData(const LatticeAddressing& addr_);
  //const LatticeAddressing& addr;
  
  LDView Host();
  LDView Device();

  SharedArray<double> rho;
  SharedArray<double> u;
  SharedArray<double> force;
  SharedArray<double> fOld;
  SharedArray<double> fNew;
};


struct Lattice {
  Lattice(int nx, int ny, int nz, double tau_s, double tau_b);
  ~Lattice();
  void Step();
  void CalcHydro();
  void InitFromHydro();
  void ZeroForce();

  SharedItem<LBParams> params;
  SharedItem<LatticeAddressing> addr;
  LatticeData data;
  /* Current timestep */
  int time_step;

};

#endif
