#include "SwimmerArray.h"
#include <math.h>

SwimmerArray* SwimmerArrayNew(int num, double hydro) {
  SwimmerArray* ans = (SwimmerArray*)malloc(sizeof(SwimmerArray));
  
  ans->h = (SwimmerArrayImpl*)malloc(sizeof(SwimmerArrayImpl));
  ans->h->num = num;
  ans->h->hydroRadius = hydro;
  
  ans->h->r = (double*)malloc(DQ_d * num*sizeof(double));
  ans->h->v = (double*)malloc(DQ_d * num*sizeof(double));
  ans->h->n = (double*)malloc(DQ_d * num*sizeof(double));
  ans->h->P = (double*)malloc(num * sizeof(double));
  ans->h->a = (double*)malloc(num * sizeof(double));
  ans->h->l = (double*)malloc(num * sizeof(double));
  
  ans->d_h = (SwimmerArrayImpl*)malloc(sizeof(SwimmerArrayImpl));

  ans->d_h->num = num;
  ans->d_h->hydroRadius = hydro;
  cudaMalloc(&ans->d_h->r, DQ_d * num*sizeof(double));
  cudaMalloc(&ans->d_h->v, DQ_d * num*sizeof(double));
  cudaMalloc(&ans->d_h->n, DQ_d * num*sizeof(double));
  cudaMalloc(&ans->d_h->P, num * sizeof(double));
  cudaMalloc(&ans->d_h->a, num * sizeof(double));
  cudaMalloc(&ans->d_h->l, num * sizeof(double));

  cudaMalloc(&ans->d, sizeof(SwimmerArrayImpl));
  cudaMemcpy(ans->d, ans->d_h, sizeof(SwimmerArrayImpl), cudaMemcpyHostToDevice);
  return ans;
}

void SwimmerArrayDel(SwimmerArray* sa) {
  cudaFree(sa->d);
  cudaFree(sa->d_h->l);
  cudaFree(sa->d_h->a);
  cudaFree(sa->d_h->P);
  cudaFree(sa->d_h->n);
  cudaFree(sa->d_h->v);
  cudaFree(sa->d_h->r);
  
  free(sa->d_h);
  
  free(sa->h->l);
  free(sa->h->a);
  free(sa->h->P);
  free(sa->h->n);
  free(sa->h->v);
  free(sa->h->r);

  free(sa->h);

  free(sa);
}

__device__ double peskin_delta(double x) {
  double abs_x = fabs(x);
  double root = -4. * x*x;
  double phi = -2.* abs_x;
  
  if (abs_x >= 2.0)
    return 0.;
  
  if (abs_x >= 1.0) {
    root += 12. * abs_x - 7.;
    phi += 5.;
    phi -= sqrt(root);
  } else {
    root += 4. * abs_x + 1;
    phi += 3.;
    phi += sqrt(root);
  }
  return 0.125 * phi;

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

__device__ void AccumulateDeltaForce(LatticeImpl* lat, double* r, double* F) {
  int* n = lat->addr->size;
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
	int ijk = (lat->addr->strides[DQ_X]*indices[DQ_X][i] +
		   lat->addr->strides[DQ_Y]*indices[DQ_Y][j] +
		   lat->addr->strides[DQ_Z]*indices[DQ_Z][k]);

	double delta3d = (deltas[DQ_X][i] *
			  deltas[DQ_Y][j] *
			  deltas[DQ_Z][k]);
	/* add force contributions */
	for (d=0; d<DQ_d; ++d) {
	  atomicAdd(lat->data->force_ptr + d*lat->addr->n + ijk,
		    delta3d * F[d]);
	}
	
      }/* k */
    }/* j */
  }/* i */
  
}


__global__ void DoSwimmerArrayAddForces(SwimmerArrayImpl* sa,
					LatticeImpl* lat) {
  int iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= sa->num) return;
  
  int a;
  /* tail end */
  /* only force is the propulsion */
  double r[DQ_d];
  double force[DQ_d];
  
  for (a=0; a<DQ_d; a++) {
    r[a] = sa->r[sa->num*a + iSwim] - sa->n[sa->num*a + iSwim] * sa->l[iSwim];
    force[a] = -sa->P[iSwim] * sa->n[sa->num*a + iSwim];
  }
  AccumulateDeltaForce(lat, r, force);
  
  /* head end */
  /* opposite force */
  for (a=0; a<DQ_d; a++) {
    r[a] = sa->r[sa->num*a + iSwim];
    force[a] *= -1.0;
  }
  
  AccumulateDeltaForce(lat, r, force);
}

void SwimmerArrayAddForces(SwimmerArray* sa, Lattice* lat) {
  const int nSwim = sa->h->num;
  const int blockSize = 512;
  const int numBlocks = (nSwim + blockSize - 1)/blockSize;

  DoSwimmerArrayAddForces<<<numBlocks, blockSize>>>(sa->d, lat->d);
}

__device__ void InterpVelocity(const LatticeImpl* lat, const double* r,
			       double* v) {
  const int* n = lat->addr->size;
  int indices[DQ_d][4];
  double deltas[DQ_d][4];
  double delta3d;
  double x, x0;
  int d;
  int i,j,k;

  /* zero the output */
  v[0] = v[1] = v[2] = 0.0;

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
	/* evaluate the delta function */
	delta3d = deltas[DQ_X][i] * deltas[DQ_Y][j] * deltas[DQ_Z][k];
	int ijk = (lat->addr->strides[DQ_X]*indices[DQ_X][i] +
		   lat->addr->strides[DQ_Y]*indices[DQ_Y][j] +
		   lat->addr->strides[DQ_Z]*indices[DQ_Z][k]);
	for (d=0; d<3; ++d) {
	  v[d] += delta3d * lat->data->force_ptr[d*lat->addr->n + ijk];
	}
      }
    }
  }
  
}

__global__ void DoSwimmerArrayMove(SwimmerArrayImpl* sa, LatticeImpl* lat) {
  /* Updates the swimmers' positions using:
   *     Rdot = v(R) + Fn_/(6 pi eta a)
   * where v(R) is the interpolated velocity at the position of the
   * swimmer, a is the radius and n_ is the orientation.
   */
  const int iSwim = threadIdx.x + blockIdx.x * blockDim.x;
  /* If we're out of range, skip */
  if (iSwim >= sa->num) return;
  
  const int* size = lat->addr->size;
  double v[DQ_d];
  double r[DQ_d] = {sa->r[sa->num*DQ_X + iSwim],
		    sa->r[sa->num*DQ_Y + iSwim],
		    sa->r[sa->num*DQ_Z + iSwim]};
  // CHECK
  const double eta = lat->params->tau_s*3;

  InterpVelocity(lat, r, v);
  
  double rDot[DQ_d];
  double rMinus[DQ_d];
  
  for (int d=0; d<DQ_d; d++) {
    rDot[d] = sa->P[iSwim] * sa->n[sa->num*d + iSwim];
    rDot[d] *= (1./sa->a[iSwim] - 1./sa->hydroRadius) / (6. * M_PI * eta);
    rDot[d] += v[d];
    
    rMinus[d] = r[d] - sa->n[sa->num*d + iSwim] * sa->l[iSwim];
  }
  
  double vMinus[DQ_d];
  InterpVelocity(lat, rMinus, vMinus);

  double nDot[DQ_d];
  for (int d=0; d<DQ_d; d++) {
    nDot[d] = v[d] - vMinus[d];
    /* now nDot = v(rPlus) - v(rMinus) */
    /* so divide by l to get the rate */
    nDot[d] /= sa->l[iSwim];
  }
  
  /* self.applyMove(lattice, rDot) */
  for (int d=0; d<DQ_d; d++) {
    sa->v[sa->num*d + iSwim] = rDot[d];
    
    /* new position */
    r[d] += rDot[d];
    /* deal with PBC */
    sa->r[sa->num*d + iSwim] = fmod(r[d] + size[d], size[d]);
  }

  /* self.applyTurn(lattice, nDot) */
  double newn[DQ_d];
  double norm = 0.0;
  for (int d=0; d<DQ_d; d++) {
    newn[d] =  sa->n[sa->num*d + iSwim] + nDot[d];
    norm += newn[d]*newn[d];
  }
  norm = sqrt(norm);
  for (int d=0; d<DQ_d; d++) {
    sa->n[sa->num*d + iSwim] = newn[d] / norm;
  }

}
void SwimmerArrayMove(SwimmerArray* sa, Lattice* lat) {
  const int nSwim = sa->h->num;
  const int blockSize = 512;
  const int numBlocks = (nSwim + blockSize - 1)/blockSize;
  
  DoSwimmerArrayMove<<<numBlocks, blockSize>>>(sa->d, lat->d);
}
