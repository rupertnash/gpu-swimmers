// -*- mode: C++; -*-
#ifndef TARGET_TARGET_H
#define TARGET_TARGET_H

#ifdef TARGET_FUNC_ATTR_H
// If we've had the function attributes included, undefine them before
// we include targetDP
#undef __targetHost__
#undef __target__
#undef __targetEntry__

#endif

#define __TDP_CPP_MODE
#include <targetDP.h>

#endif
