cimport _shared
cimport numpy as np
cimport SwimmerArray

cdef class Array:
    cdef object owner
    cdef _shared.SharedArray[double]* impl
    cdef np.ndarray view
    
    cdef Array init(self, owner, _shared.SharedArray[double]* sa)

cdef class PrngArray:
    cdef object owner
    cdef _shared.SharedArray[SwimmerArray.RandState]* impl
    cdef np.ndarray view
    
    cdef PrngArray init(self, owner, _shared.SharedArray[SwimmerArray.RandState]* sa)
