cimport SwimmerArray
cimport _shared
from lb cimport Lattice
from Lists cimport SharedVectorList

cdef class CommonParams:
    cdef adopt(self, _shared.SharedItem[SwimmerArray.CommonParams]* cp)
    cdef bint owner
    cdef _shared.SharedItem[SwimmerArray.CommonParams]* impl
    
cdef class Array:
    cpdef AddForces(self, Lattice lat)
    cpdef Move(self, Lattice lat)

    cdef SwimmerArray.SwimmerArray* impl
    cdef readonly SharedVectorList r
    cdef readonly SharedVectorList v
    cdef readonly SharedVectorList n
    cdef CommonParams common
