// -*- mode: C++; -*-
#ifndef LISTS_H
#define LISTS_H

#include "Array.h"
#include "dq.h"
#include "target/rand.h"

typedef target::rand::State RandState;

typedef Array<double, 1, 1> ScalarList;
typedef Array<double, 1, 3> VectorList;
typedef Array<RandState, 1, 1> RandList;

#endif
