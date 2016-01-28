// -*- mode: C++; -*-
#ifndef LATTICE_H
#define LATTICE_H

#include "dq.h"
#include "LBParams.h"

#include "Array.h"
#include "SharedItem.h"
#include "SharedArray.h"

typedef Array<double, DQ_d, 1> ScalarField;
typedef Array<double, DQ_d, DQ_d> VectorField;
typedef Array<double, DQ_d, DQ_q> DistField;

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
