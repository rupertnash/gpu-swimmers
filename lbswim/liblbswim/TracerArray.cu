#include "TracerArray.h"
#include <math.h>
#include "interp.cu"

TracerArray::TracerArray(const int num_) : 
  num(num_),
  r(num_*DQ_d), s(num_*DQ_d), v(num_*DQ_d)
{
}


__global__ void DoTracerArrayMove(const int nPart,
				  double* part_r,
				  double* part_s,
				  double* part_v,
				  const LatticeAddressing* addr,
				  const double* lat_u) {
  /* Updates the positions using:
   *     Rdot = v(R) 
   * where v(R) is the interpolated velocity at the position of the
   * particle
   */
  const int iPart = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iPart >= nPart) return;
  
  const int* size = addr->size;
  double v[DQ_d];
  double r[DQ_d] = {part_r[nPart*DQ_X + iPart],
		    part_r[nPart*DQ_Y + iPart],
		    part_r[nPart*DQ_Z + iPart]};

  InterpVelocity(addr, lat_u, r, v);
    
  for (int d=0; d<DQ_d; d++) {
    part_v[nPart*d + iPart] = v[d];
    r[d] += v[d];
    /* deal with PBC */
    part_r[nPart*d + iPart] = fmod(r[d] + size[d], double(size[d]));
    /* and then update the unwrapped coords */
    part_s[d] += v[d];
  }
}

void TracerArray::Move(Lattice* lat) {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;
  
  DoTracerArrayMove<<<numBlocks, BlockSize>>>(num,
					      r.device, s.device, v.device,
					      lat->addr.device,
					      lat->data->u.device);
}
