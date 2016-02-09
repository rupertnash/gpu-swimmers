// -*- mode: C++; -*-
#ifndef _TARGET_H
#define _TARGET_H
#include <targetDP.h>

#ifdef __NVCC__
#undef __targetHost__
#define __targetHost__ __host__
#endif

#ifdef HOST
#undef HOST
#endif
#define HOST __targetHost__

#define TARGET __target__
#define BOTH TARGET HOST

#endif
