// -*- mode: C++; -*-
#ifndef LISTS_H
#define LISTS_H

#include "Array.h"
#include "dq.h"
struct curandStateXORWOW;
typedef struct curandStateXORWOW RandState;

typedef Array<double, 1, 1> ScalarList;
typedef Array<double, 1, 3> VectorList;
typedef Array<RandState, 1, 1> RandList;

#endif
