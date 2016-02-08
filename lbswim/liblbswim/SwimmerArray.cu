// -*- mode: C++; -*-
#include <curand_kernel.h>
#include "SwimmerArray.h"
#include <math.h>
#include "delta.cu"
#include "interp.cu"

__global__ void DoInitPrng(const unsigned long long seed,
			   RandList& prngs) {
  const size_t n = prngs.indexer.shape[0];
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (i >= n) return;
  auto this_prng = prngs[i][0];
  curand_init(seed, i, 0, &this_prng);
}

SwimmerArray::SwimmerArray(const size_t num_, const CommonParams* p) : 
  num(num_), common(*p),
  r(array<size_t,1>{num_}), v(array<size_t,1>{num_}), n(array<size_t,1>{num_}),
  prng(array<size_t,1>{num})
{
  const size_t numBlocks = (num + BlockSize - 1)/BlockSize;
  DoInitPrng<<<numBlocks, BlockSize>>>(p->seed, prng.Device());
  // pull state to host
  prng.D2H();
}

__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
		    assumed,
		    __double_as_longlong(val + __longlong_as_double(assumed)));
    /* Note: uses integer comparison to avoid hang in case of NaN
     * (since NaN != NaN) */
  } while (assumed != old);
  
  return __longlong_as_double(old);
}

__device__ void AccumulateDeltaForce(VectorField& lat_force, const double* r, const double* F) {
  size_t indices[DQ_d][4];
  double deltas[DQ_d][4];
  double x, x0;
  size_t d;
  size_t i,j,k;
  const Shape& n = lat_force.indexer.shape;
  
  for (d=0; d<DQ_d; d++) {
    x0 = ceil(r[d]-2.);
    for (i=0; i<4; i++) {
      x = x0+i;
      indices[d][i] = ((int)x + n[d]) % n[d];
      deltas[d][i] = peskin_delta(r[d]-x);
    }
  }
  
  for (i=0; i<4; ++i) {
    auto&& plane_force = lat_force[indices[DQ_X][i]];
    for (j=0; j<4; ++j) {
      auto&& line_force = plane_force[indices[DQ_Y][j]];
      for (k=0; k<4; ++k) {
	auto&& point_force = line_force[indices[DQ_Z][k]];
	double delta3d = (deltas[DQ_X][i] *
			  deltas[DQ_Y][j] *
			  deltas[DQ_Z][k]);
	/* add force contributions */
	for (d=0; d<DQ_d; ++d) {
	  atomicAdd(&point_force[d], delta3d * F[d]);
	}
	
      }/* k */
    }/* j */
  }/* i */
  
}


__global__ void DoSwimmerArrayAddForces(const CommonParams& p,
					const VectorList& swim_r,
					const VectorList& swim_n,
					VectorField& lat_force) {
  const size_t nSwim = swim_r.indexer.shape[0];
  const size_t iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= nSwim) return;
  
  size_t a;
  /* tail end */
  /* only force is the propulsion */
  double r[DQ_d];
  double force[DQ_d];
  
  for (a=0; a<DQ_d; a++) {
    r[a] = swim_r[iSwim][a] - swim_n[iSwim][a] * p.l;
    force[a] = -p.P * swim_n[iSwim][a];
  }
  AccumulateDeltaForce(lat_force, r, force);
  
  /* head end */
  /* opposite force */
  for (a=0; a<DQ_d; a++) {
    r[a] = swim_r[iSwim][a];
    force[a] *= -1.0;
  }
  
  AccumulateDeltaForce(lat_force, r, force);
}

void SwimmerArray::AddForces(Lattice* lat) const {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;

  DoSwimmerArrayAddForces<<<numBlocks, BlockSize>>>(common.Device(),
						    r.Device(), n.Device(),
						    lat->data.force.Device());
}

__global__ void DoSwimmerArrayMove(const CommonParams& common,
				   VectorList& swim_r,
				   VectorList& swim_v,
				   VectorList& swim_n,
				   RandList& swim_prng,
				   const VectorField& lat_u) {
  /* Updates the swimmers' positions using:
   *     Rdot = v(R) + Fn_/(6 pi eta a)
   * where v(R) is the interpolated velocity at the position of the
   * swimmer, a is the radius and n_ is the orientation.
   */  
  const size_t nSwim = swim_r.indexer.shape[0];
  
  const size_t iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= nSwim) return;
  
  auto size = lat_u.indexer.shape;

  auto v = InterpVelocity(lat_u, swim_r[iSwim]);
  
  double rDot[DQ_d];
  double rMinus[DQ_d];
  
  for (int d=0; d<DQ_d; d++) {
    rDot[d] = common.mobility * common.P * swim_n[iSwim][d];

    if (!common.translational_advection_off)
      rDot[d] += v[d];
    
    rMinus[d] = swim_r[iSwim][d] - swim_n[iSwim][d] * common.l;
  }
  
  /* self.applyMove(lattice, rDot) */
  for (int d=0; d<DQ_d; d++) {
    swim_v[iSwim][d] = rDot[d];
    
    /* new position */
    swim_r[iSwim][d] += rDot[d];
    /* deal with PBC */
    swim_r[iSwim][d] = fmod(swim_r[iSwim][d] + size[d], double(size[d]));
  }
  
  double newn[DQ_d];
  double norm;
  
  float rand = curand_uniform(&swim_prng[iSwim][0]);
  if (rand < common.alpha) {
    // Tumble - i.e. pick a random unit vector
    // Pick a point from a Gaussian distribution and normalise it
    // We'll do the norm below.
    for (int d=0; d<DQ_d; d++) {
      newn[d] = curand_normal_double(&swim_prng[iSwim][0]);
      norm += newn[d]*newn[d];
    }

  } else if (!common.rotational_advection_off){
    // Normal rotation
    auto vMinus = InterpVelocity(lat_u, rMinus);

    double nDot[DQ_d];
    for (int d=0; d<DQ_d; d++) {
      nDot[d] = v[d] - vMinus[d];
      /* now nDot = v(rPlus) - v(rMinus) */
      /* so divide by l to get the rate */
      nDot[d] /= common.l;
    }
    
    /* self.applyTurn(lattice, nDot) */
    
    for (int d=0; d<DQ_d; d++) {
      newn[d] =  swim_n[iSwim][d] + nDot[d];
      norm += newn[d]*newn[d];
    }
  } else {
    // Do nothing if rotational advection is off
  }

  norm = sqrt(norm);
  for (int d=0; d<DQ_d; d++) {
    swim_n[iSwim][d] = newn[d] / norm;
  }
}

void SwimmerArray::Move(Lattice* lat) {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;
  
  DoSwimmerArrayMove<<<numBlocks, BlockSize>>>(common.Device(),
					       r.Device(), v.Device(), n.Device(),
					       prng.Device(),
					       lat->data.u.Device());
}
