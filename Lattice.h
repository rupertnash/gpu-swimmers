#ifndef DQ_Lattice_H
#define DQ_Lattice_H

#include "d3q15.h"
#include "LBParams.h"

typedef struct LatticeAddressing {
  int size[DQ_d];
  int strides[DQ_d];
  int n;
} LatticeAddressing;
  
typedef struct LatticeArrays {
  /* Pointers to the distribution function array etc
   */
  double *f_current_ptr;
  double *f_new_ptr;
  double *rho_ptr;
  double *u_ptr;
  double *force_ptr;  
} LatticeArrays;

typedef struct LatticeImpl {
  LBParams* params;
  LatticeAddressing* addr;
  LatticeArrays* data;
} LatticeImpl;

typedef struct Lattice {
  LatticeImpl* h;
  LatticeImpl* d;
  LatticeImpl* d_h;
  LatticeArrays* d_arrays;
  /* Current timestep */
  int time_step;

} Lattice;


#endif
