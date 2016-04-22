// -*- mode: C++; -*-
#ifndef TARGET_VECTOR_GNU_H
#define TARGET_VECTOR_GNU_H

#define TARGET_ASSERT_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_SIMD_PRAGMA

#endif // TARGET_VECTOR_GNU_H
