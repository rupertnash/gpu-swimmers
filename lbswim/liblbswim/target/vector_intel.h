// -*- mode: C++; -*-
#ifndef TARGET_VECTOR_INTEL_H
#define TARGET_VECTOR_INTEL_H

#define TARGET_ASSERT_ALIGNED(ptr, alignment) __assume_aligned(ptr, alignment)
#define TARGET_SIMD_PRAGMA _Pragma("simd")
namespace target {
  
}
#endif // TARGET_VECTOR_INTEL_H
