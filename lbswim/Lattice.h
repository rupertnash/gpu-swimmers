#ifndef DQ_Lattice_H
#define DQ_Lattice_H

#include "d3q15.h"
#include "LBParams.h"
#include "Shared.h"

struct LatticeAddressing {
  int size[DQ_d];
  int strides[DQ_d];
  int n;
};

// struct LatticeArrays {
//   /* Pointers to the distribution function array etc
//    */
//   double *f_current_ptr;
//   double *f_new_ptr;
//   double *rho_ptr;
//   double *u_ptr;
//   double *force_ptr;  
// };

// struct LatticeImpl {
//   LBParams* params;
//   LatticeAddressing* addr;
//   LatticeArrays* data;
// };

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
  LatticeData* data;
  /* Current timestep */
  int time_step;

};

#endif
