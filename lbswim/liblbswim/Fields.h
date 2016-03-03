// -*- mode: C++; -*-
#ifndef FIELDS_H
#define FIELDS_H

#include "NdArray.h"
#include "dq.h"

typedef NdArray<double, DQ_d, 1> ScalarField;
typedef NdArray<double, DQ_d, DQ_d> VectorField;
typedef NdArray<double, DQ_d, DQ_q> DistField;

#endif
