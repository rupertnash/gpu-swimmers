#include "TracerArray.h"
#include <cmath>
#include "interp.h"

TracerArray::TracerArray(const size_t num_) : 
  num(num_),
  r(array<size_t,1>{num_}), s(array<size_t,1>{num_}), v(array<size_t,1>{num_})
{
}


__targetEntry__ void DoTracerArrayMove(VectorList& part_r,
				       VectorList& part_s,
				       VectorList& part_v,
				       const VectorField& lat_u) {
  FOR_TLP(part_r.Shape()) {
    FOR_ILP(iPart) {
      /* Updates the positions using:
       *     Rdot = v(R) 
       * where v(R) is the interpolated velocity at the position of the
       * particle
       */  
      auto size = lat_u.indexer.shape;
      auto fluid_v = InterpVelocity(lat_u, part_r[iPart]);
      
      for (int d=0; d<DQ_d; d++) {
	part_v[iPart][d] = fluid_v[d];
	part_r[iPart][d] += fluid_v[d];
	/* deal with PBC */
	part_r[iPart][d] = std::fmod(part_r[iPart][d] + size[d], double(size[d]));
	/* and then update the unwrapped coords */
	part_s[iPart][d] += fluid_v[d];
      }
    }
  }
}

void TracerArray::Move(Lattice* lat) {
  target::launch(DoTracerArrayMove, r.Host().Shape())(r.Device(), s.Device(), v.Device(),
						    lat->data.u.Device());
}
