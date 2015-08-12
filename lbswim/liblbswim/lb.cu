#include <stdio.h>
#include <stdlib.h>

#include "lb.h"
#include "d3q15.h"
#include "cucall.h"

// This defines the LatticeEigenSet() for the D3Q15 velocity set
#include "d3q15.c"

LatticeData::LatticeData(const LatticeAddressing& addr) : rho(addr.n),
							   u(addr.n*DQ_d),
							   force(addr.n*DQ_d),
							   fOld(addr.n*DQ_q),
							   fNew(addr.n*DQ_q)
{
}
  
LDView LatticeData::Host() {
  LDView v;
  v.rho = rho.host;
  v.u = u.host;
  v.force = force.host;
  v.fOld = fOld.host;
  v.fNew = fNew.host;
  return v;
}
LDView LatticeData::Device(){
  LDView v;
  v.rho = rho.device;
  v.u = u.device;
  v.force = force.device;
  v.fOld = fOld.device;
  v.fNew = fNew.device;
  return v;
}

//template void SharedArray<double>::H2D();
//template class SharedArray<double>;
Lattice::Lattice(int nx, int ny, int nz, double tau_s, double tau_b) : time_step(0) {
  // Set up params and move to device
  params.host->tau_s = tau_s;
  params.host->tau_b = tau_b;
  LatticeEigenSet(params.host);

  params.H2D();
  
  // Set up addressing and move to device
  addr.host->size[0] = nx;
  addr.host->size[1] = ny;
  addr.host->size[2] = nz;

  addr.host->n = nx * ny * nz;

  addr.host->strides[DQ_X] = ny * nz;
  addr.host->strides[DQ_Y] = nz;
  addr.host->strides[DQ_Z] = 1;

  addr.H2D();
  
  // Set up data arrays, but don't copy
  data = new LatticeData(*addr.host);
}


Lattice::~Lattice() {
  delete data;}

__constant__ double DQ_delta[DQ_d][DQ_d] = {{1.0, 0.0, 0.0},
					    {0.0, 1.0, 0.0},
					    {0.0, 0.0, 1.0}};
__global__ void DoStep(const LBParams* params,
		       const LatticeAddressing* addr,
		       LDView data) {
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};
  
  /* loop indices for dimension */
  int a,b;
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*addr->strides[DQ_Z];
  
  const int nSites = addr->n;

  const double* fOld = data.fOld;
  const double* force = data.force;
  double* fNew = data.fNew;
  
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
      mode[m] += fOld[p*nSites + ijk] * params->mm[m][p];
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
    u[a] = (mode[DQ_mom(a)] + 0.5*force[a*nSites + ijk]) / mode[DQ_rho];
    mode[DQ_mom(a)] += force[a*nSites + ijk];
    usq += u[a]*u[a];
    uDOTf += u[a]*force[a*nSites + ijk];
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
      S[a][b] += 2.*omega_s*tau_s * (u[a]*force[b*nSites + ijk] + force[a*nSites + ijk]*u[b] - 2. * uDOTf * DQ_delta[a][b]);
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
    int destIdx[DQ_d];
    int dest_ijk = 0;
    for (a=0; a<DQ_d; a++) {
      /* This does the PBC */
      destIdx[a] = (siteIdx[a] + cp[a] + addr->size[a]) % addr->size[a];
      dest_ijk +=  destIdx[a] * addr->strides[a];
    }
    fNew[p*nSites + dest_ijk] = fPostCollision[p];
  }

}

void Lattice::Step() {
  const int* lat_size = addr.host->size;
  const int bs = 8;
  
  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;
  
  DoStep<<<num_blocks, block_shape>>>(params.device,
				      addr.device,
				      data->Device());

  /* Wait for completion */
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  /* Swap the f ptrs on the device */
  double* tmp = data->fOld.device;
  data->fOld.device = data->fNew.device;
  data->fNew.device = tmp;

  time_step++;
}


__global__ void DoCalcHydro(const LBParams* params, const LatticeAddressing* addr, LDView data) {
  /* Index for loops over dimension */
  int a;
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			     threadIdx.y + blockIdx.y * blockDim.y,
			     threadIdx.z + blockIdx.z * blockDim.z};
  
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*addr->strides[DQ_Z];
  
  const int nSites = addr->n;
  
  /* Indices for loops over dists/modes */
  int m, p;
  double mode[DQ_q];

    /* dists at time = t */
  const double* f_t = data.fOld;
  const double* force = data.force;
  double* u = data.u;
  double* rho = data.rho;

  /* compute the modes */
  for (m=0; m<DQ_q; m++) {
    mode[m] = 0.;
    for (p=0; p<DQ_q; p++) {
      mode[m] += f_t[p*nSites + ijk] * params->mm[m][p];
    }
  }
  
  rho[ijk] = mode[DQ_rho];
  
  /* Work out the site fluid velocity
   *   rho*u= (rho*u') + F*\Delta t /2
   */
  for (a=0; a<DQ_d; a++) {
    u[a*nSites + ijk] = (mode[DQ_mom(a)] + 0.5*force[a*nSites + ijk]) / mode[DQ_rho];
  }
	
}

void Lattice::CalcHydro() {
  const int* lat_size = addr.host->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoCalcHydro<<<num_blocks, block_shape>>>(params.device, addr.device, data->Device());
}

__global__ void DoInitFromHydro(const LBParams* params, const LatticeAddressing* addr, LDView data) {
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			     threadIdx.y + blockIdx.y * blockDim.y,
			     threadIdx.z + blockIdx.z * blockDim.z};
  
  /* loop indices for dimension */
  int a;
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*addr->strides[DQ_Z];
  const int nSites = addr->n;

    /* dists at time = t */
  double* f_t = data.fOld;
  //const double* force = lat->data->force_ptr;
  const double* u = data.u;
  const double* rho = data.rho;

  /* loop indices for velocities & modes */
  int p,m;
  /* Create the modes, all zeroed out */
  double mode[DQ_q];
  for (m=0; m<DQ_q; m++)
    mode[m] = 0.0;

  /* Now set the equilibrium values */
  /* Density */
  mode[DQ_rho] = rho[ijk];
  
  /* Momentum */
  for (a=0; a<DQ_d; a++) {
    mode[DQ_mom(a)] = rho[ijk] * u[a*nSites + ijk];// - 0.5*force[a*nSites + ijk];
  }

  /* Stress */
  mode[DQ_SXX] = rho[ijk] * u[DQ_X*nSites + ijk] * u[DQ_X*nSites + ijk];
  mode[DQ_SXY] = rho[ijk] * u[DQ_X*nSites + ijk] * u[DQ_Y*nSites + ijk];
  mode[DQ_SXZ] = rho[ijk] * u[DQ_X*nSites + ijk] * u[DQ_Y*nSites + ijk];
  
  mode[DQ_SYY] = rho[ijk] * u[DQ_Y*nSites + ijk] * u[DQ_Y*nSites + ijk];
  mode[DQ_SYZ] = rho[ijk] * u[DQ_Y*nSites + ijk] * u[DQ_Z*nSites + ijk];
  
  mode[DQ_SZZ] = rho[ijk] * u[DQ_Z*nSites + ijk] * u[DQ_Z*nSites + ijk];

  /* Now project modes->dists */
  for (p=0; p<DQ_q; p++) {
    f_t[p*nSites + ijk] = 0.;
    for (m=0; m<DQ_q; m++) {
      f_t[p*nSites + ijk] += mode[m] * params->mmi[p][m];
    }
  }

}

void Lattice::InitFromHydro() {
  const int* lat_size = addr.host->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoInitFromHydro<<<num_blocks, block_shape>>>(params.device, addr.device, data->Device());
}

__global__ void DoLatticeZeroForce(const LBParams* params, const LatticeAddressing* addr, LDView data) {
  /* Index for loops over dimension */
  int a;
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			     threadIdx.y + blockIdx.y * blockDim.y,
			     threadIdx.z + blockIdx.z * blockDim.z};
  
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*addr->strides[DQ_Z];
  
  const int nSites = addr->n;
  
  for (a=0; a<DQ_d; a++) {
    data.force[a*nSites + ijk] = 0.0;
  }
}

void Lattice::ZeroForce() {
  const int* lat_size = addr.host->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoLatticeZeroForce<<<num_blocks, block_shape>>>(params.device, addr.device, data->Device());
}

