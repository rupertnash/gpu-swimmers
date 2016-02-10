// -*- mode: C++; -*-

// Get the function host/target attribute macros without targetDP

#ifndef TARGET_FUNC_ATTR_H
#define TARGET_FUNC_ATTR_H

#ifndef _TDP_INCLUDED
// Only if we haven't got targetDP already

#ifdef __NVCC__

#define __targetHost__ __host__
#define __target__ __device__
#define __targetEntry__ __global__

#else

#define __targetHost__ 
#define __target__ 
#define __targetEntry__ 

#endif

#endif

// Add __targetBoth__ always
#define __targetBoth__ __target__ __targetHost__

#endif
