// -*- mode: C++; -*-
#ifndef LISTS_H
#define LISTS_H

#include "target/NdArray.h"
#include "target/rand.h"
#include "dq.h"

typedef target::rand::State RandState;

typedef target::NdArray<double, 1, 1> ScalarList;
typedef target::NdArray<double, 1, 3> VectorList;

// const size_t RSalign = alignof(RandState);
// const size_t RSsize = sizeof(RandState);

typedef target::NdArray<RandState, 1, 1> RandList;

#endif
