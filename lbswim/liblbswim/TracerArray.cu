// -*- mode: C++; -*-
#include "TracerArray.h"
#include <math.h>
#include "interp.cu"

TracerArray::TracerArray(const size_t num_) : 
  num(num_),
  r(array<size_t,1>{num_}), s(array<size_t,1>{num_}), v(array<size_t,1>{num_})
{
}


__global__ void DoTracerArrayMove(VectorList* part_r_ptr,
				  VectorList* part_s_ptr,
				  VectorList* part_v_ptr,
				  const VectorField* lat_u) {
  const size_t nPart = part_r_ptr->indexer.shape[0];
  const size_t iPart = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iPart >= nPart) return;
  
  /* Updates the positions using:
   *     Rdot = v(R) 
   * where v(R) is the interpolated velocity at the position of the
   * particle
   */
  auto part_r = (*part_r_ptr)[iPart];
  auto part_s = (*part_s_ptr)[iPart];
  auto part_v = (*part_v_ptr)[iPart];
  
  auto size = lat_u->indexer.shape;
  auto fluid_v = InterpVelocity(*lat_u, part_r);
    
  for (int d=0; d<DQ_d; d++) {
    part_v[d] = fluid_v[d];
    part_r[d] += fluid_v[d];
    /* deal with PBC */
    part_r[d] = fmod(part_r[d] + size[d], double(size[d]));
    /* and then update the unwrapped coords */
    part_s[d] += fluid_v[d];
  }
}

void TracerArray::Move(Lattice* lat) {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;
  
  DoTracerArrayMove<<<numBlocks, BlockSize>>>(r.Device(), s.Device(), v.Device(),
					      lat->data.u.Device());
}
