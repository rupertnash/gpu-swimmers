cimport TracerArray
from lb cimport Lattice
from Lists cimport SharedVectorList

cdef class Array:
    cpdef AddForces(self, Lattice lat)
    cpdef Move(self, Lattice lat)

    cdef TracerArray.TracerArray* impl
    cdef SharedVectorList r
    cdef SharedVectorList s
    cdef SharedVectorList v
