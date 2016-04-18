// -*- mode: C++ -*-
// This is the combined CUDA source file to escape linking difficulties
#include "target/target.h"

#include "SharedItem.hpp"
#include "SharedNdArray.hpp"

#include "Lattice.cxx"
#include "SwimmerArray.cxx"
#include "TracerArray.cxx"

#include "target/rand_cuda.cxx"
#include <targetDP_CUDA.c>

#include "python_template_instantiations.cxx"
