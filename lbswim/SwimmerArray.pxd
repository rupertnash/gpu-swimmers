cimport _array
from _shared cimport SharedItem
from _lb cimport Lattice
from dq cimport *
from Lists cimport _VectorList

cdef extern from "SwimmerArray.h":
    cdef cppclass RandState:
        pass
    ctypedef _array.NdArray[RandState, ONE, ONE] _RandList
    
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
        SharedItem[_VectorList] r
        SharedItem[_VectorList] v
        SharedItem[_VectorList] n
        SharedItem[_RandList] prng
