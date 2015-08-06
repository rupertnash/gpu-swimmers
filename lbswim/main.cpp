#include <stdio.h>
#include "Lattice.h"
#include "lb.h"
#include "SwimmerArray.h"

int main(int argc, char* argv[]) {
  int size = 16;
  double tau = 0.25;
  
  Lattice lat(size,size,size, tau,tau);
  const LatticeAddressing* addr = lat.addr.host;
  
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
	lat.data->rho.host[ijk] = d;
	
	for (int a=0; a<DQ_d; a++) {
	  lat.data->u.host[a*addr->n + ijk] = 0.0;
	  lat.data->force.host[a*addr->n + ijk] = 0.0;
	}
	
      }
    }
  }
  
  /* Copy to device */
  lat.data->rho.H2D();
  lat.data->u.H2D();
  lat.data->force.H2D();

  /* Set the dists */
  lat.InitFromHydro();
  
  SwimmerArray swimmers(2, 1.5);
  swimmers.r.host[0] = 1.0; swimmers.r.host[1] = 4.0;
  swimmers.r.host[2] = 1.0; swimmers.r.host[1] = 4.0;
  swimmers.r.host[4] = 1.0; swimmers.r.host[1] = 4.0;
  
  while (lat.time_step < 100) {
    lat.Step();
    if (lat.time_step % 10 == 0) {
      lat.CalcHydro();
      lat.data->rho.D2H();
      int i = 8, j = 8;
      for (int k=0; k<size; k++) {
	int ijk = addr->strides[DQ_X]*i + 
	  addr->strides[DQ_Y]*j + 
	  addr->strides[DQ_Z]*k;
	
	printf(" %f", lat.data->rho.host[ijk]);
      }
      printf("\n");
    }
  }
  
}
