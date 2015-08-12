#ifndef INTERP_CU
#define INTERP_CU

#include "Lattice.h"

#include "delta.cu"

__device__ void InterpVelocity(const LatticeAddressing* addr,
			       const double* lat_u,
			       const double* r,
			       double* v) {
  const int* n = addr->size;
  int indices[DQ_d][4];
  double deltas[DQ_d][4];
  double delta3d;
  double x, x0;
  int d;
  int i,j,k;

  /* zero the output */
  v[0] = v[1] = v[2] = 0.0;

  for (d=0; d<DQ_d; d++) {
    x0 = ceil(r[d]-2.);
    for (i=0; i<4; i++) {
      x = x0+i;
      indices[d][i] = ((int)x + n[d]) % n[d];
      deltas[d][i] = peskin_delta(r[d]-x);
    }
  }

  for (i=0; i<4; ++i) {
    for (j=0; j<4; ++j) {
      for (k=0; k<4; ++k) {
	/* evaluate the delta function */
	delta3d = deltas[DQ_X][i] * deltas[DQ_Y][j] * deltas[DQ_Z][k];
	int ijk = (addr->strides[DQ_X]*indices[DQ_X][i] +
		   addr->strides[DQ_Y]*indices[DQ_Y][j] +
		   addr->strides[DQ_Z]*indices[DQ_Z][k]);
	for (d=0; d<3; ++d) {
	  v[d] += delta3d * lat_u[d*addr->n + ijk];
	}
      }
    }
  }
  
}

#endif
