#ifndef DELTA_CU
#define DELTA_CU

#include <math.h>

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

#endif
