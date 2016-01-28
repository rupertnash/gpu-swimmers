// -*- mode: C++; -*-
#ifndef INTERP_CU
#define INTERP_CU

#include "Lattice.h"
#include "delta.cu"
#include "array.h"

template<typename Indexable>
__device__ array<double,DQ_d> InterpVelocity(const VectorField& lat_u,
			       Indexable r) {
  auto n = lat_u.indexer.shape;
  int indices[DQ_d][4];
  double deltas[DQ_d][4];
  double delta3d;
  double x, x0;
  int d;
  int i,j,k;
  /* zero the output */
  array<double,DQ_d> v = {0,0,0};
  //v[0] = v[1] = v[2] = 0.0;

  for (d=0; d<DQ_d; d++) {
    x0 = ceil(r[d]-2.);
    for (i=0; i<4; i++) {
      x = x0+i;
      indices[d][i] = ((int)x + n[d]) % n[d];
      deltas[d][i] = peskin_delta(r[d]-x);
    }
  }

  for (i=0; i<4; ++i) {
    auto plane_u = lat_u[indices[DQ_X][i]];
    for (j=0; j<4; ++j) {
      auto line_u = plane_u[indices[DQ_Y][j]];
      for (k=0; k<4; ++k) {
	auto point_u = line_u[indices[DQ_Z][k]];
	/* evaluate the delta function */
	delta3d = deltas[DQ_X][i] * deltas[DQ_Y][j] * deltas[DQ_Z][k];
	for (d=0; d<3; ++d) {
	  v[d] += delta3d * point_u[d];
	}
      }
    }
  }
  return v;
}

#endif
