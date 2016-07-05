#include <stdio.h>
#include <stdlib.h>
#include <utility>

#include "Lattice.h"
#include "d3q15.h"
#include "target/targetpp.h"

// This defines the LatticeEigenSet() for the D3Q15 velocity set
#include "d3q15.cxx"

LatticeData::LatticeData(const Shape& shape) : rho(shape),
					       u(shape),
					       force(shape),
					       fOld(shape),
					       fNew(shape)
{
}
  
LDView LatticeData::Host() {
  LDView v;
  v.rho = &rho.Host();
  v.u = &u.Host();
  v.force = &force.Host();
  v.fOld = &fOld.Host();
  v.fNew = &fNew.Host();
  return v;
}
LDView LatticeData::Device(){
  LDView v;
  v.rho = rho.Device();
  v.u = u.Device();
  v.force = force.Device();
  v.fOld = fOld.Device();
  v.fNew = fNew.Device();
  return v;
}

Lattice::Lattice(const Shape& shape_, double tau_s, double tau_b) : shape(shape_), data(shape_), time_step(0) {
  // Set up params and move to device
  params.Host().tau_s = tau_s;
  params.Host().tau_b = tau_b;
  LatticeEigenSet(params.Host());

  params.H2D();
  
}


Lattice::~Lattice() {
  
}

__targetConst__ double DQ_delta[DQ_d][DQ_d] = {{1.0, 0.0, 0.0},
					       {0.0, 1.0, 0.0},
					       {0.0, 0.0, 1.0}};

TARGET_KERNEL_DECLARE(LatticeStepK, 3, TARGET_DEFAULT_VVL, const LBParams*, LDView);
TARGET_KERNEL_DEFINE(LatticeStepK, const LBParams* params, LDView data) {
  auto shape = data.rho->Shape();
  FOR_TLP(threadCtx) {
    const auto fOld = threadCtx.GetCurrentElements(data.fOld);
    const auto force = threadCtx.GetCurrentElements(data.force);
    typedef target::NdArray<double, 1, DQ_q> DistList;
    DistList fPostCollision(VecLen());

    FOR_ILP(i) {
      double mode[DQ_q];
      /* convenience vars */
      double S[DQ_d][DQ_d];
      double u[DQ_d];
      double usq, TrS, uDOTf;
      
      const double tau_s = params->tau_s;
      const double tau_b = params->tau_b;
      const double omega_s = 1.0 / (tau_s + 0.5);
      const double omega_b = 1.0 / (tau_b + 0.5);
      
      /* compute the modes */
      for (size_t m=0; m<DQ_q; m++) {
	mode[m] = 0.;
	for (size_t p=0; p<DQ_q; p++) {
	  mode[m] += fOld(i, p) * params->mm[m][p];
	}
      }
      
      /* Work out the site fluid velocity
       *   rho*u= (rho*u') + F*\Delta t /2
       * and the coefficient for the momentum modes.
       *    = (rho*u') * F* \Delta t
       * (and u squared)
       */
      usq = 0.;
      uDOTf = 0.;
      for (size_t a=0; a<DQ_d; a++) {
	u[a] = (mode[DQ_mom(a)] + 0.5*force(i, a)) / mode[DQ_rho];
	mode[DQ_mom(a)] += force(i, a);
	usq += u[a]*u[a];
	uDOTf += u[a]*force(i, a);
      }
      
      /* For unequal relax trace & traceless part at different rates.
       * Equilibrium trace = rho*usq */
      
      /* First copy the stress to a convenience var */
      S[DQ_X][DQ_X] = mode[DQ_SXX];
      S[DQ_X][DQ_Y] = mode[DQ_SXY];
      S[DQ_X][DQ_Z] = mode[DQ_SXZ];
      
      S[DQ_Y][DQ_X] = mode[DQ_SXY];
      S[DQ_Y][DQ_Y] = mode[DQ_SYY];
      S[DQ_Y][DQ_Z] = mode[DQ_SYZ];
      
      S[DQ_Z][DQ_X] = mode[DQ_SXZ];
      S[DQ_Z][DQ_Y] = mode[DQ_SYZ];
      S[DQ_Z][DQ_Z] = mode[DQ_SZZ];
      
      /* Form the trace part */
      TrS = 0.;
      for (size_t a=0; a<DQ_d; a++) {
	TrS += S[a][a];
      }
      /* And the traceless part */
      for (size_t a=0; a<DQ_d; a++) {
	S[a][a] -= TrS/DQ_d;
      }
      
      /* relax the trace */
      TrS -= omega_b*(TrS - mode[DQ_rho]*usq);
      /* Add forcing part to trace */
      TrS += 2.*omega_b*tau_b * uDOTf;
      
      /* and the traceless part */
      for (size_t a=0; a<DQ_d; a++) {
	for (size_t b=0; b<DQ_d; b++) {
	  S[a][b] -= omega_s*(S[a][b] - 
			      mode[DQ_rho]*(u[a]*u[b] -usq*DQ_delta[a][b]));
	  
	  /* including traceless force */
	  S[a][b] += 2.*omega_s*tau_s * (u[a]*force(i, b) + force(i, a)*u[b] - 2. * uDOTf * DQ_delta[a][b]);
	}
	/* add the trace back on */
	S[a][a] += TrS / DQ_d;
      }
      
      /* copy S back into modes[] */
      mode[DQ_SXX] = S[DQ_X][DQ_X];
      mode[DQ_SXY] = S[DQ_X][DQ_Y];
      mode[DQ_SXZ] = S[DQ_X][DQ_Z];
      
      mode[DQ_SYY] = S[DQ_Y][DQ_Y];
      mode[DQ_SYZ] = S[DQ_Y][DQ_Z];
      
      mode[DQ_SZZ] = S[DQ_Z][DQ_Z];
      
      /* Ghosts are relaxed to zero immediately */
      mode[DQ_chi1] = 0.;
      mode[DQ_jchi1X] = 0.;
      mode[DQ_jchi1Y] = 0.;
      mode[DQ_jchi1Z] = 0.;
      mode[DQ_chi2] = 0.;
    
      /* project back to the velocity basis */
      for (size_t p=0; p<DQ_q; p++) {
	fPostCollision(i, p) = 0.;
	for (size_t m=0; m<DQ_q; m++) {
	  fPostCollision(i, p) += mode[m] * params->mmi[p][m];
	}
      }
    }

    
    FOR_ILP(siteIdx) {
      /* Stream */
      const Shape myIdx = threadCtx.GetNdIndex(siteIdx);
      for (size_t p=0; p<DQ_q; p++) {
  	const int* cp = params->ci[p];
  	Shape destIdx;
  	for (size_t a=0; a<DQ_d; a++) {
  	  /* This does the PBC */
  	  destIdx[a] = (myIdx[a] + cp[a] + shape[a]) % shape[a];
  	}
  	(*data.fNew)[destIdx][p] = fPostCollision[p][siteIdx];
      }
    
    }
  }
}

void Lattice::Step() {
  // Call the target kernel
  auto k = LatticeStepK(shape);
  k(params.Device(), data.Device());

  // Wait for completion
  target::synchronize();

  // Swap the distributions
  std::swap(data.fOld, data.fNew);
  
  time_step++;
}

TARGET_KERNEL_DECLARE(LatticeCalcHydroK, 3, TARGET_DEFAULT_VVL, const LBParams*, LDView);
TARGET_KERNEL_DEFINE(LatticeCalcHydroK, const LBParams* params, LDView data) {
  FOR_TLP(threadCtx) {
    /* dists at time = t */
    auto f_t = threadCtx.GetCurrentElements(data.fOld);
    auto force = threadCtx.GetCurrentElements(data.force);
    auto u = threadCtx.GetCurrentElements(data.u);
    auto rho = threadCtx.GetCurrentElements(data.rho);
    FOR_ILP(siteIdx) {
      /* Indices for loops over dists/modes */
      size_t m, p;
      double mode[DQ_q];
      
      /* compute the modes */
      for (m=0; m<DQ_q; m++) {
  	mode[m] = 0.;
  	for (p=0; p<DQ_q; p++) {
  	  mode[m] += f_t[siteIdx][p] * params->mm[m][p];
  	}
      }
  
      rho[siteIdx][0] = mode[DQ_rho];
  
      /* Work out the site fluid velocity
       *   rho*u= (rho*u') + F*\Delta t /2
       */
      for (size_t a=0; a<DQ_d; a++) {
  	u[siteIdx][a] = (mode[DQ_mom(a)] + 0.5*force[siteIdx][a]) / mode[DQ_rho];
      }
    }
  }
}

void Lattice::CalcHydro() {
  LatticeCalcHydroK k(shape);
  k(params.Device(), data.Device());
  target::synchronize();
}

TARGET_KERNEL_DECLARE(LatticeInitFromHydroK, 3, TARGET_DEFAULT_VVL, const LBParams*, LDView);
TARGET_KERNEL_DEFINE(LatticeInitFromHydroK, const LBParams* params, LDView data) {
  FOR_TLP(threadCtx) {
    /* dists at time = t */
    auto f_t = threadCtx.GetCurrentElements(data.fOld);
    auto u = threadCtx.GetCurrentElements(data.u);
    auto rho = threadCtx.GetCurrentElements(data.rho);

    FOR_ILP(index) {
      /* Create the modes, all zeroed out */
      double mode[DQ_q];
      for (size_t m = 0; m < DQ_q; m++)
	mode[m] = 0;
      
      /* Now set the equilibrium values */
      /* Density */
      mode[DQ_rho] = rho[index][0];
      
      /* Momentum */
      for (size_t a=0; a<DQ_d; a++) {
  	mode[DQ_mom(a)] = rho[index][0] * u[index][a];// - 0.5*force[a*nSites + ijk];
      }
      
      /* Stress */
      mode[DQ_SXX] = rho[index][0] * u[index][DQ_X] * u[index][DQ_X];
      mode[DQ_SXY] = rho[index][0] * u[index][DQ_X] * u[index][DQ_Y];
      mode[DQ_SXZ] = rho[index][0] * u[index][DQ_X] * u[index][DQ_Z];
  
      mode[DQ_SYY] = rho[index][0] * u[index][DQ_Y] * u[index][DQ_Y];
      mode[DQ_SYZ] = rho[index][0] * u[index][DQ_Y] * u[index][DQ_Z];
  
      mode[DQ_SZZ] = rho[index][0] * u[index][DQ_Z] * u[index][DQ_Z];

      /* Now project modes->dists */
      for (size_t p=0; p<DQ_q; p++) {
  	f_t[index][p] = 0.;
  	for (size_t m=0; m<DQ_q; m++) {
  	  f_t[index][p] += mode[m] * params->mmi[p][m];
  	}
      }
    }
  }
}

void Lattice::InitFromHydro() {
  LatticeInitFromHydroK k(shape);
  k(params.Device(), data.Device());
  target::synchronize();
}


TARGET_KERNEL_DECLARE(LatticeZeroForceK, 3, TARGET_DEFAULT_VVL, const LBParams*, LDView);
TARGET_KERNEL_DEFINE(LatticeZeroForceK, const LBParams* params, LDView data) {
  FOR_TLP(threadCtx) {
    auto force_v = threadCtx.GetCurrentElements(data.force);
    for (auto a = 0; a <DQ_d; ++a) {
      FOR_ILP(i) {
	force_v[i][a] = 0;
      }
    }
  }
}

void Lattice::ZeroForce() {
  auto launcher = LatticeZeroForceK(shape);
  launcher(params.Device(), data.Device());
  target::synchronize();
}

