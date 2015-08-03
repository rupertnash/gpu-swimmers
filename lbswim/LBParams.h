#ifndef LBPARAMS_H
#define LBPARAMS_H

typedef struct LBParams {
  /* speed of sound, squared */
  double cs2;
  /* quadrature weights */
  double w[DQ_q];
  /* velocities - as doubles */
  double xi[DQ_q][DQ_d];
  /* velocities - as ints */
  int ci[DQ_q][DQ_d];
  
  /* Kinetic projector, Q_i_ab = xi_i_a * xi_i_b - delta_ab cs^2 */
  double Q[DQ_q][DQ_d][DQ_d];
  
  int complement[DQ_q];
  
  /* The modes' normalizers */
  double norms[DQ_q];
  /* Matrix of eigenvectors (they are the rows) */
  double mm[DQ_q][DQ_q];
  /* and its inverse */
  double mmi[DQ_q][DQ_q];

  /* The relaxation time */
  double tau_s; /* shear relaxation time */
  double tau_b; /* bulk relaxation time */
  
} LBParams;

#endif
