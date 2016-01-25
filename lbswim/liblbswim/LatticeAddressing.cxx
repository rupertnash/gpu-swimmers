#include "LatticeAddressing.h"
#include "dq.h"

LatticeAddressing::LatticeAddressing(int nx, int ny, int nz) {
  size[DQ_X] = nx;
  size[DQ_Y] = ny;
  size[DQ_Z] = nz;

  n = nx * ny * nz;

  strides[DQ_X] = ny * nz;
  strides[DQ_Y] = nz;
  strides[DQ_Z] = 1;
}
