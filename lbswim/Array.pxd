from dq cimport *
cimport _array
cimport numpy as np

cdef class Array:
    cdef readonly np.ndarray data
    pass
