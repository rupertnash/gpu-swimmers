from _array cimport VectorList, RandList
from _shared cimport SharedItem
from _lb cimport Lattice

cdef extern from "SwimmerArray.h":
    cdef cppclass RandState:
        pass
    cdef cppclass CommonParams:
        double P
        double l
        double alpha
        double mobility
        bint translational_advection_off
        bint rotational_advection_off
        unsigned long long seed
        
        pass
    cdef cppclass SwimmerArray:
        SwimmerArray(int n, CommonParams* p)
        void AddForces(Lattice* lat)
        void Move(Lattice* lat)
        int num
        SharedItem[CommonParams] common
        SharedItem[VectorList] r
        SharedItem[VectorList] v
        SharedItem[VectorList] n
        SharedItem[RandList] prng
