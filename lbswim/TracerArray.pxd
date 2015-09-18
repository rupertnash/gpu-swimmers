from _shared cimport SharedArray, SharedItem
from _lb cimport Lattice

cdef extern from "TracerArray.h":
    cdef cppclass TracerArray:
        TracerArray(int n)
        void Move(Lattice* lat)
        int num
        SharedArray[double] r
        SharedArray[double] s
        SharedArray[double] v
