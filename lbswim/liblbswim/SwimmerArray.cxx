#include <cmath>

#include "SwimmerArray.h"
#include "interp.h"
#include "target/targetpp.h"
#include "target/atomic.h"

TARGET_KERNEL_DECLARE(InitPrngK, 1, TARGET_DEFAULT_VVL, const unsigned long long, RandList*);
TARGET_KERNEL_DEFINE(InitPrngK, const unsigned long long seed,
		     RandList* prngs) {
  FOR_TLP(thread) {
    auto prng_vec = thread.GetCurrentElements(prngs);
    FOR_ILP(i) {
      target::rand::init(seed, thread.GetNdIndex(i)[0], &prng_vec[i][0]);
    }
  }
}

SwimmerArray::SwimmerArray(const size_t num_, const CommonParams* p) : 
  num(num_), common(*p),
  r(target::array<size_t,1>{num_}),
  v(target::array<size_t,1>{num_}),
  n(target::array<size_t,1>{num_}),
  prng(target::array<size_t,1>{num})
{
  InitPrngK k(prng.Host().Shape());
  k(p->seed, prng.Device());
  target::synchronize();
  // pull state to host
  prng.D2H();
}


__target__ void AccumulateDeltaForce(VectorField* lat_force_ptr, const double* r, const double* F) {
  size_t indices[DQ_d][4];
  double deltas[DQ_d][4];
  double x, x0;
  size_t d;
  size_t i,j,k;
  VectorField& lat_force = *lat_force_ptr;
  const Shape& n = lat_force.Shape();
  
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
	  target::atomic::increment(point_force[d], delta3d * F[d]);
	}
	
      }/* k */
    }/* j */
  }/* i */
  
}

TARGET_KERNEL_DECLARE(SwimmerArrayAddForcesK, 1, TARGET_DEFAULT_VVL,
		      const CommonParams*, const VectorList*, const VectorList*, VectorField*);
TARGET_KERNEL_DEFINE(SwimmerArrayAddForcesK,
		     const CommonParams* p, const VectorList* swim_r, const VectorList* swim_n,
		     VectorField* lat_force) {
  FOR_TLP(thread) {
    auto swim_r_vec = thread.GetCurrentElements(swim_r);
    auto swim_n_vec = thread.GetCurrentElements(swim_n);
    FOR_ILP(iSwim) {  
      size_t a;
      /* tail end */
      /* only force is the propulsion */
      double r[DQ_d];
      double force[DQ_d];
  
      for (a=0; a<DQ_d; a++) {
	r[a] = swim_r_vec[iSwim][a] - swim_n_vec[iSwim][a] * p->l;
	force[a] = -p->P * swim_n_vec[iSwim][a];
      }
      AccumulateDeltaForce(lat_force, r, force);
  
      /* head end */
      /* opposite force */
      for (a=0; a<DQ_d; a++) {
	r[a] = swim_r_vec[iSwim][a];
	force[a] *= -1.0;
      }
  
      AccumulateDeltaForce(lat_force, r, force);
    }
  }
}

void SwimmerArray::AddForces(Lattice* lat) const {
  SwimmerArrayAddForcesK k(r.Host().Shape());
  k(common.Device(),
    r.Device(), n.Device(),
    lat->data.force.Device());
  target::synchronize();
}
TARGET_KERNEL_DECLARE(SwimmerArrayMoveK, 1, TARGET_DEFAULT_VVL,
		      const CommonParams*,
		      VectorList*,
		      VectorList*,
		      VectorList*,
		      RandList*,
		      const VectorField*);
TARGET_KERNEL_DEFINE(SwimmerArrayMoveK, const CommonParams* common,
					VectorList* swim_r,
					VectorList* swim_v,
					VectorList* swim_n,
					RandList* swim_prng,
					const VectorField* lat_u) {
  /* Updates the swimmers' positions using:
   *     Rdot = v(R) + Fn_/(6 pi eta a)
   * where v(R) is the interpolated velocity at the position of the
   * swimmer, a is the radius and n_ is the orientation.
   */  
  FOR_TLP(threadCtx) {
    
    auto swim_r_vec = threadCtx.GetCurrentElements(swim_r);
    auto swim_v_vec = threadCtx.GetCurrentElements(swim_v);
    auto swim_n_vec = threadCtx.GetCurrentElements(swim_n);
    auto swim_prng_vec = threadCtx.GetCurrentElements(swim_prng);
    
    FOR_ILP(iSwim) {
  
      auto size = lat_u->Shape();

      auto v = InterpVelocity(lat_u, swim_r_vec[iSwim]);
  
      double rDot[DQ_d];
      double rMinus[DQ_d];
  
      for (int d=0; d<DQ_d; d++) {
	rDot[d] = common->mobility * common->P * swim_n_vec[iSwim][d];

	if (!common->translational_advection_off)
	  rDot[d] += v[d];
    
	rMinus[d] = swim_r_vec[iSwim][d] - swim_n_vec[iSwim][d] * common->l;
      }
  
      /* self.applyMove(lattice, rDot) */
      for (int d=0; d<DQ_d; d++) {
	swim_v_vec[iSwim][d] = rDot[d];
    
	/* new position */
	swim_r_vec[iSwim][d] += rDot[d];
	/* deal with PBC */
	swim_r_vec[iSwim][d] = fmod(swim_r_vec[iSwim][d] + size[d], double(size[d]));
      }
  
      double newn[DQ_d];
      double norm;
  
      float rand = target::rand::uniform(&swim_prng_vec[iSwim][0]);
      if (rand < common->alpha) {
	// Tumble - i.e. pick a random unit vector
	// Pick a point from a Gaussian distribution and normalise it
	// We'll do the norm below.
	for (int d=0; d<DQ_d; d++) {
	  newn[d] = target::rand::normal_double(&swim_prng_vec[iSwim][0]);
	  norm += newn[d]*newn[d];
	}

      } else if (!common->rotational_advection_off){
	// Normal rotation
	auto vMinus = InterpVelocity(lat_u, rMinus);

	double nDot[DQ_d];
	for (int d=0; d<DQ_d; d++) {
	  nDot[d] = v[d] - vMinus[d];
	  /* now nDot = v(rPlus) - v(rMinus) */
	  /* so divide by l to get the rate */
	  nDot[d] /= common->l;
	}
    
	/* self.applyTurn(lattice, nDot) */
    
	for (int d=0; d<DQ_d; d++) {
	  newn[d] =  swim_n_vec[iSwim][d] + nDot[d];
	  norm += newn[d]*newn[d];
	}
      } else {
	// Do nothing if rotational advection is off
      }

      norm = sqrt(norm);
      for (int d=0; d<DQ_d; d++) {
	swim_n_vec[iSwim][d] = newn[d] / norm;
      }
    }
  }
}

void SwimmerArray::Move(Lattice* lat) {
  SwimmerArrayMoveK k(r.Host().Shape());
  k(common.Device(),
    r.Device(), v.Device(), n.Device(),
    prng.Device(),
    lat->data.u.Device());
  target::synchronize();
}
