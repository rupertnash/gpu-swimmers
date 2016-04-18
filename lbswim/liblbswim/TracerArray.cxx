#include "TracerArray.h"
#include <cmath>
#include "interp.h"
#include "target/targetpp.h"

TracerArray::TracerArray(const size_t num_) : 
  num(num_),
  r(array<size_t,1>{num_}), s(array<size_t,1>{num_}), v(array<size_t,1>{num_})
{
}

TARGET_KERNEL_DECLARE(TracerArrayMoveK, 1, VVL,
		      VectorList*,
		      VectorList*,
		      VectorList*,
		      const VectorField*);
TARGET_KERNEL_DEFINE(TracerArrayMoveK, VectorList* part_r,
				       VectorList* part_s,
				       VectorList* part_v,
				       const VectorField* lat_u) {
  FOR_TLP(thread) {
    auto part_r_vec = thread.GetCurrentElements(part_r);
    auto part_s_vec = thread.GetCurrentElements(part_s);
    auto part_v_vec = thread.GetCurrentElements(part_v);
    
    FOR_ILP(iPart, thread) {
      /* Updates the positions using:
       *     Rdot = v(R) 
       * where v(R) is the interpolated velocity at the position of the
       * particle
       */  
      auto size = lat_u->indexer.shape;
      auto fluid_v = InterpVelocity(lat_u, part_r_vec[iPart]);
      
      for (int d=0; d<DQ_d; d++) {
	part_v_vec[iPart][d] = fluid_v[d];
	part_r_vec[iPart][d] += fluid_v[d];
	/* deal with PBC */
	part_r_vec[iPart][d] = std::fmod(part_r_vec[iPart][d] + size[d], double(size[d]));
	/* and then update the unwrapped coords */
	part_s_vec[iPart][d] += fluid_v[d];
      }
    }
  }
}

void TracerArray::Move(Lattice* lat) {
  TracerArrayMoveK k(r.Host().Shape());
  k(r.Device(), s.Device(), v.Device(),
    lat->data.u.Device());
  target::synchronize();
}
