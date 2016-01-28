// -*- mode: C++; -*-
#include <stdio.h>
#include <stdlib.h>

#include "Lattice.h"
#include "d3q15.h"
#include "cucall.h"

// This defines the LatticeEigenSet() for the D3Q15 velocity set
#include "d3q15.c"

LatticeData::LatticeData(const Shape& shape) : rho(shape),
					       u(shape),
					       force(shape),
					       fOld(shape),
					       fNew(shape)
{
}
  
LDView LatticeData::Host() {
  LDView v;
  v.rho = rho.Host();
  v.u = u.Host();
  v.force = force.Host();
  v.fOld = fOld.Host();
  v.fNew = fNew.Host();
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

//template void SharedArray<double>::H2D();
//template class SharedArray<double>;
Lattice::Lattice(const Shape& shape_, double tau_s, double tau_b) : shape(shape_), data(shape_), time_step(0) {
  // Set up params and move to device
  params->tau_s = tau_s;
  params->tau_b = tau_b;
  LatticeEigenSet(*params);

  params.H2D();
  
  // Set up data arrays, but don't copy
  // data = new LatticeData(*addr);
}


Lattice::~Lattice() {
  
}

__constant__ double DQ_delta[DQ_d][DQ_d] = {{1.0, 0.0, 0.0},
					    {0.0, 1.0, 0.0},
					    {0.0, 0.0, 1.0}};
__global__ void DoStep(const LBParams* params, LDView data) {
  
  const Shape siteIdx = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};

  const Shape& shape = data.rho->indexer.shape;
  /* loop indices for dimension */
  int a,b;
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= shape[a])
      return;
  }

  auto fOld = (*data.fOld)[siteIdx];
  auto force = (*data.force)[siteIdx];
  
  /* loop indices for velocities & modes */
  int p,m;
  
  double mode[DQ_q];
  /* convenience vars */
  double S[DQ_d][DQ_d];
  double u[DQ_d];
  double usq, TrS, uDOTf;
  
  const double tau_s = params->tau_s;
  const double tau_b = params->tau_b;
  const double omega_s = 1.0 / (tau_s + 0.5);
  const double omega_b = 1.0 / (tau_b + 0.5);

  /* Post collision dists */
  double fPostCollision[DQ_q];

  /* compute the modes */
  for (m=0; m<DQ_q; m++) {
    mode[m] = 0.;
    for (p=0; p<DQ_q; p++) {
      mode[m] += fOld[p] * params->mm[m][p];
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
  for (a=0; a<DQ_d; a++) {
    u[a] = (mode[DQ_mom(a)] + 0.5*force[a]) / mode[DQ_rho];
    mode[DQ_mom(a)] += force[a];
    usq += u[a]*u[a];
    uDOTf += u[a]*force[a];
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
  for (a=0; a<DQ_d; a++) {
    TrS += S[a][a];
  }
  /* And the traceless part */
  for (a=0; a<DQ_d; a++) {
    S[a][a] -= TrS/DQ_d;
  }
  
  /* relax the trace */
  TrS -= omega_b*(TrS - mode[DQ_rho]*usq);
  /* Add forcing part to trace */
  TrS += 2.*omega_b*tau_b * uDOTf;
  
  /* and the traceless part */
  for (a=0; a<DQ_d; a++) {
    for (b=0; b<DQ_d; b++) {
      S[a][b] -= omega_s*(S[a][b] - 
			  mode[DQ_rho]*(u[a]*u[b] -usq*DQ_delta[a][b]));
      
      /* including traceless force */
      S[a][b] += 2.*omega_s*tau_s * (u[a]*force[b] + force[a]*u[b] - 2. * uDOTf * DQ_delta[a][b]);
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
  for (p=0; p<DQ_q; p++) {
   fPostCollision[p] = 0.;
    for (m=0; m<DQ_q; m++) {
      fPostCollision[p] += mode[m] * params->mmi[p][m];
    }
  }

  /* Stream */
  for (p=0; p<DQ_q; p++) {
    const int* cp = params->ci[p];
    Shape destIdx;
    for (a=0; a<DQ_d; a++) {
      /* This does the PBC */
      destIdx[a] = (siteIdx[a] + cp[a] + shape[a]) % shape[a];
    }
    (*data.fNew)[destIdx][p] = fPostCollision[p];
  }

}

void Lattice::Step() {
  //const int* lat_size = addr->size;
  const int bs = 8;
  
  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (shape[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (shape[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (shape[DQ_Z] + block_shape.z - 1)/block_shape.z;
  
  DoStep<<<num_blocks, block_shape>>>(params.Device(),
				      //addr.Device(),
				      data.Device());

  /* Wait for completion */
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  /* Swap the f ptrs on the device */
  SharedItem<DistField>::SwapDevicePointers(data.fOld, data.fNew);

  time_step++;
}


__global__ void DoCalcHydro(const LBParams* params, LDView data) {
  const Shape siteIdx = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};

  const Shape& shape = data.rho->indexer.shape;
  /* If we're out of bounds, skip */
  for (size_t a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= shape[a])
      return;
  }

  /* Indices for loops over dists/modes */
  size_t m, p;
  double mode[DQ_q];

  /* dists at time = t */
  auto f_t = const_cast<const DistField*>(data.fOld)->operator[](siteIdx);
  auto force = const_cast<const VectorField*>(data.force)->operator[](siteIdx);
  auto u = data.u->operator[](siteIdx);
  auto rho = data.rho->operator[](siteIdx);

  /* compute the modes */
  for (m=0; m<DQ_q; m++) {
    mode[m] = 0.;
    for (p=0; p<DQ_q; p++) {
      mode[m] += f_t[p] * params->mm[m][p];
    }
  }
  
  rho[0] = mode[DQ_rho];
  
  /* Work out the site fluid velocity
   *   rho*u= (rho*u') + F*\Delta t /2
   */
  for (size_t a=0; a<DQ_d; a++) {
    u[a] = (mode[DQ_mom(a)] + 0.5*force[a]) / mode[DQ_rho];
  }
	
}

void Lattice::CalcHydro() {
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (shape[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (shape[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (shape[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoCalcHydro<<<num_blocks, block_shape>>>(params.Device(), data.Device());
}

__global__ void DoInitFromHydro(const LBParams* params, LDView data) {
  const Shape siteIdx = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};

  const Shape& shape = data.rho->indexer.shape;
  /* If we're out of bounds, skip */
  for (size_t a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= shape[a])
      return;
  }

  /* dists at time = t */
  auto f_t = data.fOld->operator[](siteIdx);
  auto u = const_cast<const VectorField*>(data.u)->operator[](siteIdx);
  auto rho = const_cast<const ScalarField*>(data.rho)->operator[](siteIdx)[0];

  /* Indices for loops over dists/modes */
  size_t m, p;
  /* Create the modes, all zeroed out */
  array<double, DQ_q> mode;

  /* Now set the equilibrium values */
  /* Density */
  mode[DQ_rho] = rho;
  
  /* Momentum */
  for (size_t a=0; a<DQ_d; a++) {
    mode[DQ_mom(a)] = rho * u[a];// - 0.5*force[a*nSites + ijk];
  }

  /* Stress */
  mode[DQ_SXX] = rho * u[DQ_X] * u[DQ_X];
  mode[DQ_SXY] = rho * u[DQ_X] * u[DQ_Y];
  mode[DQ_SXZ] = rho * u[DQ_X] * u[DQ_Z];
  
  mode[DQ_SYY] = rho * u[DQ_Y] * u[DQ_Y];
  mode[DQ_SYZ] = rho * u[DQ_Y] * u[DQ_Z];
  
  mode[DQ_SZZ] = rho * u[DQ_Z] * u[DQ_Z];

  /* Now project modes->dists */
  for (p=0; p<DQ_q; p++) {
    f_t[p] = 0.;
    for (m=0; m<DQ_q; m++) {
      f_t[p] += mode[m] * params->mmi[p][m];
    }
  }

}

void Lattice::InitFromHydro() {
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (shape[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (shape[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (shape[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoInitFromHydro<<<num_blocks, block_shape>>>(params.Device(), data.Device());
}

__global__ void DoLatticeZeroForce(const LBParams* params, LDView data) {
  const Shape siteIdx = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};

  const Shape& shape = data.rho->indexer.shape;
  /* If we're out of bounds, skip */
  for (size_t a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= shape[a])
      return;
  }

  /* dists at time = t */
  auto force = data.force->operator[](siteIdx);
  
  for (size_t a=0; a<DQ_d; a++) {
    force[a] = 0.0;
  }
}

void Lattice::ZeroForce() {
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (shape[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (shape[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (shape[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoLatticeZeroForce<<<num_blocks, block_shape>>>(params.Device(), data.Device());
}

