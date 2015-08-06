#ifndef D3Q15_H
#define D3Q15_H

#define DQ_d 3
#define DQ_q 15

/* dimensions */
#define DQ_X 0
#define DQ_Y 1
#define DQ_Z 2

/* mode names */
#define DQ_rho 0
#define DQ_momX 1
#define DQ_momY 2
#define DQ_momZ 3
#define DQ_SXX 4
#define DQ_SXY 5
#define DQ_SXZ 6
#define DQ_SYY 7
#define DQ_SYZ 8
#define DQ_SZZ 9
#define DQ_chi1 10
#define DQ_jchi1X 11
#define DQ_jchi1Y 12
#define DQ_jchi1Z 13
#define DQ_chi2 14
#define DQ_mom(dim) 1+dim

#include "LBParams.h"

void LatticeEigenSet(LBParams* lat);

#endif
