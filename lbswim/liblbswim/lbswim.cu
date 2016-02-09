// -*- mode: C++ -*-
// This is the combined CUDA source file to escape linking difficulties
#include "targetDP_CUDA.c"

#include "SharedItem.hpp"
#include "SharedArray.hpp"

#include "Lattice.cxx"
#include "SwimmerArray.cxx"
#include "TracerArray.cxx"

template class SharedItem<ScalarList>;
template class SharedItem<VectorList>;

template class SharedItem<ScalarField>;
template class SharedItem<VectorField>;
template class SharedItem<DistField>;

template class SharedItem<RandList>;

template class SharedItem<CommonParams>;
template class SharedItem<LBParams>;
