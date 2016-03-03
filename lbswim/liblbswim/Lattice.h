// -*- mode: C++; -*-
#ifndef LATTICE_H
#define LATTICE_H

#include "dq.h"
#include "LBParams.h"

#include "NdArray.h"
#include "SharedItem.h"
#include "SharedNdArray.h"

#include "Fields.h"

struct LDView {
  ScalarField* rho;
  VectorField* u;
  VectorField* force;
  DistField* fOld;
  DistField* fNew;
};

struct LatticeData {
  LatticeData(const Shape& shape_);
  
  LDView Host();
  LDView Device();

  SharedItem<ScalarField> rho;
  SharedItem<VectorField> u;
  SharedItem<VectorField> force;
  SharedItem<DistField> fOld;
  SharedItem<DistField> fNew;
};


struct Lattice {
  Lattice(const Shape& shape, double tau_s, double tau_b);
  ~Lattice();
  void Step();
  void CalcHydro();
  void InitFromHydro();
  void ZeroForce();

  SharedItem<LBParams> params;
  Shape shape;
  LatticeData data;
  /* Current timestep */
  int time_step;

};

#endif
