#include <curand_kernel.h>
#include "SwimmerArray.h"
#include <math.h>
#include "delta.cu"
#include "interp.cu"

__global__ void DoInitPrng(const int nSwim,
			   const unsigned long long seed,
			   curandStateXORWOW* prngs) {
  const int iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= nSwim) return;
  curand_init(seed, iSwim, 0, prngs+iSwim);
}

SwimmerArray::SwimmerArray(const int num_, const CommonParams* p) : 
  num(num_), common(p),
  r(num_*DQ_d), v(num_*DQ_d), n(num_*DQ_d),
  prng(num)
{
  const int numBlocks = (num + BlockSize - 1)/BlockSize;
  DoInitPrng<<<numBlocks, BlockSize>>>(num, p->seed, prng.device);
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

__device__ void AccumulateDeltaForce(const LatticeAddressing* addr, double* lat_force, double* r, double* F) {
  const int* n = addr->size;
  int indices[DQ_d][4];
  double deltas[DQ_d][4];
  double x, x0;
  int d;
  int i,j,k;

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
	int ijk = (addr->strides[DQ_X]*indices[DQ_X][i] +
		   addr->strides[DQ_Y]*indices[DQ_Y][j] +
		   addr->strides[DQ_Z]*indices[DQ_Z][k]);

	double delta3d = (deltas[DQ_X][i] *
			  deltas[DQ_Y][j] *
			  deltas[DQ_Z][k]);
	/* add force contributions */
	for (d=0; d<DQ_d; ++d) {
	  atomicAdd(lat_force + d * addr->n + ijk,
		    delta3d * F[d]);
	}
	
      }/* k */
    }/* j */
  }/* i */
  
}


__global__ void DoSwimmerArrayAddForces(const int nSwim,
					const CommonParams* p,
					const double* swim_r,
					const double* swim_n,
					const LatticeAddressing* addr,
					double* lat_force) {
  const int iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= nSwim) return;
  
  int a;
  /* tail end */
  /* only force is the propulsion */
  double r[DQ_d];
  double force[DQ_d];
  
  for (a=0; a<DQ_d; a++) {
    r[a] = swim_r[nSwim*a + iSwim] - swim_n[nSwim*a + iSwim] * p->l;
    force[a] = -p->P * swim_n[nSwim*a + iSwim];
  }
  AccumulateDeltaForce(addr, lat_force, r, force);
  
  /* head end */
  /* opposite force */
  for (a=0; a<DQ_d; a++) {
    r[a] = swim_r[nSwim*a + iSwim];
    force[a] *= -1.0;
  }
  
  AccumulateDeltaForce(addr, lat_force, r, force);
}

void SwimmerArray::AddForces(Lattice* lat) const {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;

  DoSwimmerArrayAddForces<<<numBlocks, BlockSize>>>(num,
						    common.device,
						    r.device, n.device,
						    lat->addr.device, lat->data->force.device);
}

__global__ void DoSwimmerArrayMove(const int nSwim,
				   const CommonParams* common,
				   double* swim_r,
				   double* swim_v,
				   double* swim_n,
				   RandState* swim_prng,
				   const LatticeAddressing* addr,
				   const double* lat_u) {
  /* Updates the swimmers' positions using:
   *     Rdot = v(R) + Fn_/(6 pi eta a)
   * where v(R) is the interpolated velocity at the position of the
   * swimmer, a is the radius and n_ is the orientation.
   */
  const int iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= nSwim) return;
  
  const int* size = addr->size;
  double v[DQ_d];
  double r[DQ_d] = {swim_r[nSwim*DQ_X + iSwim],
		    swim_r[nSwim*DQ_Y + iSwim],
		    swim_r[nSwim*DQ_Z + iSwim]};

  InterpVelocity(addr, lat_u, r, v);
  
  double rDot[DQ_d];
  double rMinus[DQ_d];
  
  for (int d=0; d<DQ_d; d++) {
    rDot[d] = common->mobility * common->P * swim_n[nSwim*d + iSwim];

    if (!common->translational_advection_off)
      rDot[d] += v[d];
    
    rMinus[d] = r[d] - swim_n[nSwim*d + iSwim] * common->l;
  }
  
  /* self.applyMove(lattice, rDot) */
  for (int d=0; d<DQ_d; d++) {
    swim_v[nSwim*d + iSwim] = rDot[d];
    
    /* new position */
    r[d] += rDot[d];
    /* deal with PBC */
    swim_r[nSwim*d + iSwim] = fmod(r[d] + size[d], double(size[d]));
  }
  
  double newn[DQ_d];
  double norm;
  
  float rand = curand_uniform(swim_prng + iSwim);
  if (rand < common->alpha) {
    // Tumble - i.e. pick a random unit vector
    // Pick a point from a Gaussian distribution and normalise it
    // We'll do the norm below.
    for (int d=0; d<DQ_d; d++) {
      newn[d] = curand_normal_double(swim_prng + iSwim);
      norm += newn[d]*newn[d];
    }

  } else if (!common->rotational_advection_off){
    // Normal rotation
    double vMinus[DQ_d];
    InterpVelocity(addr, lat_u, rMinus, vMinus);
    
    double nDot[DQ_d];
    for (int d=0; d<DQ_d; d++) {
      nDot[d] = v[d] - vMinus[d];
      /* now nDot = v(rPlus) - v(rMinus) */
      /* so divide by l to get the rate */
      nDot[d] /= common->l;
    }
    
    /* self.applyTurn(lattice, nDot) */
    
    for (int d=0; d<DQ_d; d++) {
      newn[d] =  swim_n[nSwim*d + iSwim] + nDot[d];
      norm += newn[d]*newn[d];
    }
  } else {
    // Do nothing if rotational advection is off
  }

  norm = sqrt(norm);
  for (int d=0; d<DQ_d; d++) {
    swim_n[nSwim*d + iSwim] = newn[d] / norm;
  }
}

void SwimmerArray::Move(Lattice* lat) {
  const int numBlocks = (num + BlockSize - 1)/BlockSize;
  
  DoSwimmerArrayMove<<<numBlocks, BlockSize>>>(num, common.device,
					       r.device, v.device, n.device,
					       prng.device,
					       lat->addr.device,
					       lat->data->u.device);
}
