// -*- mode: C++; -*-
#ifndef LISTS_H
#define LISTS_H

#include "NdArray.h"
#include "dq.h"
#include "target/rand.h"

typedef target::rand::State RandState;

typedef NdArray<double, 1, 1> ScalarList;
typedef NdArray<double, 1, 3> VectorList;
typedef NdArray<RandState, 1, 1> RandList;

#endif
