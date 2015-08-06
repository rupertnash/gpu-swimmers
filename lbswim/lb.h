#ifndef LB_H
#define LB_H

#include "Lattice.h"

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
