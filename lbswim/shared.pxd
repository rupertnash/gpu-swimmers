cimport _shared
cimport numpy as np

cdef class Array:
    cdef object owner
    cdef _shared.SharedArray[double]* impl
    cdef np.ndarray view
    
    cdef Array init(self, owner, _shared.SharedArray[double]* sa)
