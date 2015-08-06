from _shared cimport SharedArray
from _lb cimport Lattice

cdef extern from "lbswim/SwimmerArray.h":
    cdef cppclass SwimmerArray:
        SwimmerArray(int n, double hydro)
        void AddForces(Lattice* lat)
        void Move(Lattice* lat)
        int num
        double hydroRadius
        SharedArray[double] r
        SharedArray[double] v
        SharedArray[double] n
        SharedArray[double] P
        SharedArray[double] a
        SharedArray[double] l

