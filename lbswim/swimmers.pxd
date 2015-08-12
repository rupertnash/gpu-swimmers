cimport SwimmerArray
cimport shared
cimport _shared
from lb cimport Lattice

cdef class CommonParams:
    cdef adopt(self, _shared.SharedItem[SwimmerArray.CommonParams]* cp)
    cdef bint owner
    cdef _shared.SharedItem[SwimmerArray.CommonParams]* impl
    
cdef class Array:
    cpdef AddForces(self, Lattice lat)
    cpdef Move(self, Lattice lat)

    cdef SwimmerArray.SwimmerArray* impl
    cdef shared.Array _r
    cdef shared.Array _v
    cdef shared.Array _n
    cdef CommonParams _common
