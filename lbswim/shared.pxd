cimport _shared
cimport _array
cimport numpy as np
cimport SwimmerArray

cdef class ScalarList:
    cdef object owner
    cdef _shared.SharedItem[_array.ScalarList]* impl
    cdef np.ndarray view
    
    cdef ScalarList init(self, owner, _shared.SharedItem[_array.ScalarList]* sa)

cdef class VectorList:
    cdef object owner
    cdef _shared.SharedItem[_array.VectorList]* impl
    cdef np.ndarray view
    
    cdef VectorList init(self, owner, _shared.SharedItem[_array.VectorList]* sa)

cdef class RandList:
    cdef object owner
    cdef _shared.SharedItem[_array.RandList]* impl
    cdef np.ndarray view
    
    cdef RandList init(self, owner, _shared.SharedItem[_array.RandList]* sa)

cdef class ScalarField:
    cdef object owner
    cdef _shared.SharedItem[_array.ScalarField]* impl
    cdef np.ndarray view
    
    cdef ScalarField init(self, owner, _shared.SharedItem[_array.ScalarField]* sa)

cdef class VectorField:
    cdef object owner
    cdef _shared.SharedItem[_array.VectorField]* impl
    cdef np.ndarray view
    
    cdef VectorField init(self, owner, _shared.SharedItem[_array.VectorField]* sa)

cdef class DistField:
    cdef object owner
    cdef _shared.SharedItem[_array.DistField]* impl
    cdef np.ndarray view
    
    cdef DistField init(self, owner, _shared.SharedItem[_array.DistField]* sa)
