#include <stdio.h>
#include "Lattice.h"
#include "lb.h"
#include "SwimmerArray.h"

int main(int argc, char* argv[]) {
  int size = 16;
  double tau = 0.25;
  
  Lattice* lat = LatticeNew(size,size,size, tau,tau);
  LatticeAddressing* addr = lat->h->addr;
  
  /* Set the initial conditions:
   * u = 0
   * rho = 1 except a peak in the box centre
   */
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      for (int k=0; k<size; k++) {
	int ijk = addr->strides[DQ_X]*i + 
	      addr->strides[DQ_Y]*j + 
	      addr->strides[DQ_Z]*k;
	
	double d = (i==8 && j==8 && k == 8) ? 1.1 : 1.0;
	
	lat->h->data->rho_ptr[ijk] = d;
	
	for (int a=0; a<DQ_d; a++) {
	  lat->h->data->u_ptr[a*addr->n + ijk] = 0.0;
	  lat->h->data->force_ptr[a*addr->n + ijk] = 0.0;
	}
	
      }
    }
  }
  
  /* Copy to device */
  LatticeRhoH2D(lat);
  LatticeUH2D(lat);
  LatticeForceH2D(lat);
  /* Set the dists */
  LatticeInitFromHydro(lat);
  
  SwimmerArray* swimmers = SwimmerArrayNew(2, 1.5);
  swimmers->h->r[0] = 1.0; swimmers->h->r[1] = 4.0;
  swimmers->h->r[2] = 1.0; swimmers->h->r[1] = 4.0;
  swimmers->h->r[4] = 1.0; swimmers->h->r[1] = 4.0;
  
  while (lat->time_step < 100) {
    LatticeStep(lat);
    if (lat->time_step % 10 == 0) {
      LatticeCalcHydro(lat);
      LatticeRhoD2H(lat);
      int i = 8, j = 8;
      for (int k=0; k<size; k++) {
	int ijk = addr->strides[DQ_X]*i + 
	  addr->strides[DQ_Y]*j + 
	  addr->strides[DQ_Z]*k;
	
	printf(" %f", lat->h->data->rho_ptr[ijk]);
      }
      printf("\n");
    }
  }
  
}
