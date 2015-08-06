from SwimmerArray cimport SwimmerArray
cimport shared
from lb cimport Lattice

cdef class Array:
    cpdef AddForces(self, Lattice lat)
    cpdef Move(self, Lattice lat)

    cdef SwimmerArray* impl
    cdef shared.Array _r
    cdef shared.Array _v
    cdef shared.Array _n
    cdef shared.Array _P
    cdef shared.Array _a
    cdef shared.Array _l
