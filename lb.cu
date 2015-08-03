#include <stdio.h>
#include <stdlib.h>

#include "lb.h"
#include "d3q15.h"

// This defines the LatticeEigenSet() for the D3Q15 velocity set
#include "d3q15.c"

#define CUDA_SAFE_CALL(call) do {					\
  cudaError err = call;							\
  if (err != cudaSuccess) {						\
    fprintf(stderr, "CUDA error in file '%s' on line %i: %s\n",		\
	    __FILE__, __LINE__, cudaGetErrorString(err));		\
    exit(EXIT_FAILURE);							\
  }									\
  } while(0)
  
LatticeImpl* LatticeInitHost(int nx, int ny, int nz, double tau_s, double tau_b) {
  LatticeImpl* lat = (LatticeImpl*)malloc(sizeof(LatticeImpl));

  /* LB parameters */
  lat->params = (LBParams*)malloc(sizeof(LBParams));
  LatticeEigenSet(lat->params);
  lat->params->tau_s = tau_s;
  lat->params->tau_b = tau_b;
  
  /* Addressing parameters */
  lat->addr = (LatticeAddressing*)malloc(sizeof(LatticeAddressing));
  
  lat->addr->size[0] = nx;
  lat->addr->size[1] = ny;
  lat->addr->size[2] = nz;
  
  lat->addr->n = nx * ny * nz;
  lat->addr->strides[DQ_X] = ny * nz;
  lat->addr->strides[DQ_Y] = nz;
  lat->addr->strides[DQ_Z] = 1;
  
  /* Data arrays */
  lat->data = (LatticeArrays*)malloc(sizeof(LatticeArrays));

  lat->data->f_current_ptr = (double *)malloc(lat->addr->n * DQ_q * sizeof(double));
  lat->data->f_new_ptr = (double *)malloc(lat->addr->n * DQ_q * sizeof(double));
  lat->data->rho_ptr = (double *)malloc(lat->addr->n * sizeof(double));
  lat->data->u_ptr = (double *)malloc(lat->addr->n * DQ_d * sizeof(double));
  
  lat->data->force_ptr = (double *)malloc(lat->addr->n * DQ_d * sizeof(double));
  
  return lat;
}


Lattice* LatticeNew(int nx, int ny, int nz, double tau_s, double tau_b) {
  Lattice *lat = (Lattice *)malloc(sizeof(Lattice));
  lat->time_step = 0;
  
  /* Create the host structure */
  lat->h = LatticeInitHost(nx, ny, nz, tau_s, tau_b);

  LatticeAddressing* addr = lat->h->addr;
  
  /* Alloc the host's pointers to the device structs */
  lat->d_h = (LatticeImpl*)malloc(sizeof(LatticeImpl));

  /* Alloc and copy the parameters on the device */
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->params, sizeof(LBParams)));
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->params,
			    lat->h->params, sizeof(LBParams), cudaMemcpyHostToDevice));

  /* Alloc and copy the addressing to the device */
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->addr, sizeof(LatticeAddressing)));
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->addr,
			    addr, sizeof(LatticeAddressing), cudaMemcpyHostToDevice));
  
  /* Alloc the host's pointers to the device data arrays */
  lat->d_h->data = (LatticeArrays*)malloc(sizeof(LatticeArrays));
  /* Alloc the device data arrays */
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->data->f_current_ptr,
			    addr->n * DQ_q * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->data->f_new_ptr,
			    addr->n * DQ_q * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->data->rho_ptr,
			    addr->n * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->data->u_ptr,
			    addr->n * DQ_d * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_h->data->force_ptr,
			    addr->n * DQ_d * sizeof(double)));

  /* Alloc and copy the pointers to the device arrays to the device */
  CUDA_SAFE_CALL(cudaMalloc(&lat->d_arrays, sizeof(LatticeArrays)));
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_arrays,
			    lat->d_h->data, sizeof(LatticeArrays), cudaMemcpyHostToDevice));

  /* Alloc and setup on the device the whole lot */
  LatticeImpl devCopy;
  devCopy.params = lat->d_h->params;
  devCopy.addr = lat->d_h->addr;
  devCopy.data = lat->d_arrays;
  
  CUDA_SAFE_CALL(cudaMalloc(&lat->d, sizeof(LatticeImpl)));
  CUDA_SAFE_CALL(cudaMemcpy(lat->d, &devCopy,
			    sizeof(LatticeImpl), cudaMemcpyHostToDevice));

  return lat;
}


void LatticeDel(Lattice* lat) {
  CUDA_SAFE_CALL(cudaFree(lat->d));
  CUDA_SAFE_CALL(cudaFree(lat->d_arrays));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->data->f_current_ptr));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->data->f_new_ptr));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->data->rho_ptr));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->data->u_ptr));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->data->force_ptr));
  free(lat->d_h->data);
  CUDA_SAFE_CALL(cudaFree(lat->d_h->addr));
  CUDA_SAFE_CALL(cudaFree(lat->d_h->params));
  free(lat->d_h);

  free(lat->h->data->f_current_ptr);
  free(lat->h->data->f_new_ptr);
  free(lat->h->data->rho_ptr);
  free(lat->h->data->u_ptr);
  free(lat->h->data->force_ptr);

  free(lat->h->data);
  
  free(lat->h->addr);
  free(lat->h->params);
  free(lat->h);

  free(lat);

}

__constant__ double DQ_delta[DQ_d][DQ_d] = {{1.0, 0.0, 0.0},
					    {0.0, 1.0, 0.0},
					    {0.0, 0.0, 1.0}};
__global__ void DoStep(LatticeImpl* lat) {
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};
  
  /* loop indices for dimension */
  int a,b;
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= lat->addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*lat->addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*lat->addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*lat->addr->strides[DQ_Z];
  
  const int nSites = lat->addr->n;
  
  /* loop indices for velocities & modes */
  int p,m;
  
  double mode[DQ_q];
  /* convenience vars */
  double S[DQ_d][DQ_d];
  double u[DQ_d];
  double usq, TrS, uDOTf;
  
  const double tau_s = lat->params->tau_s;
  const double tau_b = lat->params->tau_b;
  const double omega_s = 1.0 / (tau_s + 0.5);
  const double omega_b = 1.0 / (tau_b + 0.5);

  /* dists at time = t */
  const double* f_t = lat->data->f_current_ptr;
  /* Post collision dists */
  double fPostCollision[DQ_q];
  /* dists at time = t + 1*/
  double* f_tp1 = lat->data->f_new_ptr;

  const double* force = lat->data->force_ptr;

  /* compute the modes */
  for (m=0; m<DQ_q; m++) {
    mode[m] = 0.;
    for (p=0; p<DQ_q; p++) {
      mode[m] += f_t[p*nSites + ijk] * lat->params->mm[m][p];
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
      fPostCollision[p] += mode[m] * lat->params->mmi[p][m];
    }
  }

  /* Stream */
  for (p=0; p<DQ_q; p++) {
    const int* cp = lat->params->ci[p];
    int destIdx[DQ_d];
    int dest_ijk = 0;
    for (a=0; a<DQ_d; a++) {
      /* This does the PBC */
      destIdx[a] = (siteIdx[a] + cp[a] + lat->addr->size[a]) % lat->addr->size[a];
      dest_ijk +=  destIdx[a] * lat->addr->strides[a];
    }
    f_tp1[p*nSites + dest_ijk] = fPostCollision[p];
  }

}

/* Make sure to launch only once! */
__global__ void DoSwapDists(LatticeImpl* lat) {
  double* tmp = lat->data->f_current_ptr;
  lat->data->f_current_ptr = lat->data->f_new_ptr;
  lat->data->f_new_ptr = tmp;
}

void LatticeStep(Lattice* lat) {
  const int* lat_size = lat->h->addr->size;
  const int bs = 8;
  
  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoStep<<<num_blocks, block_shape>>>(lat->d);

  /* Wait for completion */
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  /* Swap the f ptrs on the device */
  DoSwapDists<<<1,1>>>(lat->d);
  
  lat->time_step++;
}


__global__ void DoCalcHydro(LatticeImpl *lat) {
  /* Index for loops over dimension */
  int a;
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			     threadIdx.y + blockIdx.y * blockDim.y,
			     threadIdx.z + blockIdx.z * blockDim.z};
  
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= lat->addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*lat->addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*lat->addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*lat->addr->strides[DQ_Z];
  
  const int nSites = lat->addr->n;
  
  /* Indices for loops over dists/modes */
  int m, p;
  double mode[DQ_q];

    /* dists at time = t */
  const double* f_t = lat->data->f_current_ptr;
  const double* force = lat->data->force_ptr;
  double* u = lat->data->u_ptr;
  double* rho = lat->data->rho_ptr;

  /* compute the modes */
  for (m=0; m<DQ_q; m++) {
    mode[m] = 0.;
    for (p=0; p<DQ_q; p++) {
      mode[m] += f_t[p*nSites + ijk] * lat->params->mm[m][p];
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

void LatticeCalcHydro(Lattice* lat) {
  const int* lat_size = lat->h->addr->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoCalcHydro<<<num_blocks, block_shape>>>(lat->d);
}

__global__ void DoInitFromHydro(LatticeImpl* lat) {
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			 threadIdx.y + blockIdx.y * blockDim.y,
			 threadIdx.z + blockIdx.z * blockDim.z};
  
  /* loop indices for dimension */
  int a;
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= lat->addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*lat->addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*lat->addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*lat->addr->strides[DQ_Z];
  const int nSites = lat->addr->n;

    /* dists at time = t */
  double* f_t = lat->data->f_current_ptr;
  //const double* force = lat->data->force_ptr;
  const double* u = lat->data->u_ptr;
  const double* rho = lat->data->rho_ptr;

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
      f_t[p*nSites + ijk] += mode[m] * lat->params->mmi[p][m];
    }
  }

}

void LatticeInitFromHydro(Lattice* lat) {
  const int* lat_size = lat->h->addr->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoInitFromHydro<<<num_blocks, block_shape>>>(lat->d);
}

__global__ void DoLatticeZeroForce(LatticeImpl* lat) {
  /* Index for loops over dimension */
  int a;
  const int siteIdx[DQ_d] = {threadIdx.x + blockIdx.x * blockDim.x,
			     threadIdx.y + blockIdx.y * blockDim.y,
			     threadIdx.z + blockIdx.z * blockDim.z};
  
  /* If we're out of bounds, skip */
  for (a=0; a<DQ_d; a++) {
    if (siteIdx[a] >= lat->addr->size[a])
      return;
  }
  const int ijk = siteIdx[DQ_X]*lat->addr->strides[DQ_X] + 
    siteIdx[DQ_Y]*lat->addr->strides[DQ_Y] + 
    siteIdx[DQ_Z]*lat->addr->strides[DQ_Z];
  
  const int nSites = lat->addr->n;
  
  for (a=0; a<DQ_d; a++) {
    lat->data->force_ptr[a*nSites + ijk] = 0.0;
  }
}

void LatticeZeroForce(Lattice* lat) {
  const int* lat_size = lat->h->addr->size;
  const int bs = 8;

  dim3 block_shape;
  block_shape.x = bs;
  block_shape.y = bs;
  block_shape.z = bs;

  dim3 num_blocks;
  num_blocks.x = (lat_size[DQ_X] + block_shape.x - 1)/block_shape.x;
  num_blocks.y = (lat_size[DQ_Y] + block_shape.y - 1)/block_shape.y;
  num_blocks.z = (lat_size[DQ_Z] + block_shape.z - 1)/block_shape.z;

  DoLatticeZeroForce<<<num_blocks, block_shape>>>(lat->d);
}

void LatticeRhoH2D(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->data->rho_ptr,
			    lat->h->data->rho_ptr,
			    addr->n * sizeof(double),
			    cudaMemcpyHostToDevice));
}

void LatticeRhoD2H(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->h->data->rho_ptr,
			    lat->d_h->data->rho_ptr,
			    addr->n * sizeof(double),
			    cudaMemcpyDeviceToHost));
}

void LatticeUH2D(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->data->u_ptr,
			    lat->h->data->u_ptr,
			    addr->n * DQ_d * sizeof(double),
			    cudaMemcpyHostToDevice));
}

void LatticeUD2H(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->h->data->u_ptr,
			    lat->d_h->data->u_ptr,
			    addr->n * DQ_d * sizeof(double),
			    cudaMemcpyDeviceToHost));
}

void LatticeForceH2D(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->data->force_ptr,
			    lat->h->data->force_ptr,
			    addr->n * DQ_d * sizeof(double),
			    cudaMemcpyHostToDevice));
}

void LatticeForceD2H(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->h->data->force_ptr,
			    lat->d_h->data->force_ptr,
			    addr->n * DQ_d * sizeof(double),
			    cudaMemcpyDeviceToHost));
}

void LatticeFH2D(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->d_h->data->f_current_ptr,
			    lat->h->data->f_current_ptr,
			    addr->n * DQ_q * sizeof(double),
			    cudaMemcpyHostToDevice));
}

void LatticeFD2H(Lattice* lat) {
  LatticeAddressing* addr = lat->h->addr;
  CUDA_SAFE_CALL(cudaMemcpy(lat->h->data->f_current_ptr,
			    lat->d_h->data->f_current_ptr,
			    addr->n * DQ_q * sizeof(double),
			    cudaMemcpyDeviceToHost));
}

void LatticeH2D(Lattice* lat) {
  LatticeFH2D(lat);
  LatticeRhoH2D(lat);
  LatticeUH2D(lat);
  LatticeForceH2D(lat);
}

void LatticeD2H(Lattice* lat) {
  LatticeFD2H(lat);
  LatticeRhoD2H(lat);
  LatticeUD2H(lat);
  LatticeForceD2H(lat);
}
