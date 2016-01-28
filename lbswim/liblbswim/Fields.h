// -*- mode: C++; -*-
#ifndef FIELDS_H
#define FIELDS_H

#include "Array.h"
#include "dq.h"

typedef Array<double, DQ_d, 1> ScalarField;
typedef Array<double, DQ_d, DQ_d> VectorField;
typedef Array<double, DQ_d, DQ_q> DistField;

#endif
