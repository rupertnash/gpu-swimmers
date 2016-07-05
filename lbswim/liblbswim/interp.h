// -*- mode: C++; -*-
#ifndef INTERP_H
#define INTERP_H

#include <cmath>

#include "Lattice.h"
#include "target/array.h"

template<typename Float>
__target__ Float peskin_delta(Float x) {
  Float abs_x = std::fabs(x);
  Float root = -4. * x*x;
  Float phi = -2.* abs_x;
  
  if (abs_x >= 2.0)
    return 0.;
  
  if (abs_x >= 1.0) {
    root += 12. * abs_x - 7.;
    phi += 5.;
    phi -= std::sqrt(root);
  } else {
    root += 4. * abs_x + 1;
    phi += 3.;
    phi += std::sqrt(root);
  }
  return 0.125 * phi;
}

template<typename Indexable>
__target__ target::array<double,DQ_d> InterpVelocity(const VectorField* lat_u_ptr,
			       const Indexable& r) {
  auto& lat_u = *lat_u_ptr;
  auto n = lat_u.Shape();
  
  int indices[DQ_d][4];
  double deltas[DQ_d][4];
  double delta3d;
  double x, x0;
  int d;
  int i,j,k;
  /* zero the output */
  target::array<double,DQ_d> v = {0,0,0};
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
