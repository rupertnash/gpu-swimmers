// -*- mode: C++; -*-
#ifndef LATTICE_H
#define LATTICE_H

#include "dq.h"
#include "LBParams.h"

#include "target/NdArray.h"
#include "target/SharedItem.h"
#include "target/SharedNdArray.h"

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

  target::SharedItem<ScalarField> rho;
  target::SharedItem<VectorField> u;
  target::SharedItem<VectorField> force;
  target::SharedItem<DistField> fOld;
  target::SharedItem<DistField> fNew;
};


struct Lattice {
  Lattice(const Shape& shape, double tau_s, double tau_b);
  ~Lattice();
  void Step();
  void CalcHydro();
  void InitFromHydro();
  void ZeroForce();

  target::SharedItem<LBParams> params;
  Shape shape;
  LatticeData data;
  /* Current timestep */
  int time_step;

};

#endif
