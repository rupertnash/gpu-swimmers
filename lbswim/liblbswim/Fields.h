// -*- mode: C++; -*-
#ifndef FIELDS_H
#define FIELDS_H

#include "target/NdArray.h"
#include "dq.h"

typedef target::NdArray<double, DQ_d, 1> ScalarField;
typedef target::NdArray<double, DQ_d, DQ_d> VectorField;
typedef target::NdArray<double, DQ_d, DQ_q> DistField;

#endif
