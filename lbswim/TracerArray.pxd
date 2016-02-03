from _shared cimport SharedItem
from Lists cimport _VectorList
from _lb cimport Lattice

cdef extern from "TracerArray.h":
    cdef cppclass TracerArray:
        TracerArray(int n)
        void Move(Lattice* lat)
        int num
        SharedItem[_VectorList] r
        SharedItem[_VectorList] s
        SharedItem[_VectorList] v
