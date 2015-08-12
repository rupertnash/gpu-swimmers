cimport TracerArray
cimport shared
from lb cimport Lattice
    
cdef class Array:
    cpdef AddForces(self, Lattice lat)
    cpdef Move(self, Lattice lat)

    cdef TracerArray.TracerArray* impl
    cdef shared.Array _r
    cdef shared.Array _v
