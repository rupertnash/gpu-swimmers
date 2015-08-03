#ifndef LB_H
#define LB_H

#include "Lattice.h"

Lattice* LatticeNew(int nx, int ny, int nz, double tau_s, double tau_b);
void LatticeDel(Lattice* lat);
void LatticeStep(Lattice* lat);
void LatticeCalcHydro(Lattice* lat);
void LatticeInitFromHydro(Lattice* lat);

void LatticeZeroForce(Lattice* lat);

void LatticeRhoH2D(Lattice* lat);
void LatticeRhoD2H(Lattice* lat);

void LatticeUH2D(Lattice* lat);
void LatticeUD2H(Lattice* lat);

void LatticeForceH2D(Lattice* lat);
void LatticeForceD2H(Lattice* lat);

void LatticeFH2D(Lattice* lat);
void LatticeFD2H(Lattice* lat);

void LatticeH2D(Lattice* lat);
void LatticeD2H(Lattice* lat);

#endif
