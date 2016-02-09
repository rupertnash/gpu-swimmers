from dq cimport *
cimport _array
cimport Array
cimport _shared
cimport numpy as np

ctypedef _array.Array[double, ONE, ONE] _ScalarList
ctypedef _array.Array[double, ONE, DQ_d] _VectorList

cdef class ScalarList(Array.Array):
    cdef _array.ArrayHelper[_ScalarList]* impl
    cdef ScalarList Init(ScalarList self, _ScalarList& impl)

cdef class SharedScalarList(ScalarList):
    cdef _shared.SharedItem[_ScalarList]* shared_impl
    cdef SharedScalarList ShInit(SharedScalarList self, _shared.SharedItem[_ScalarList]* impl)
    cpdef void H2D(SharedScalarList self)
    cpdef void D2H(SharedScalarList self)

cdef class VectorList(Array.Array):
    cdef _array.ArrayHelper[_VectorList]* impl
    cdef VectorList Init(VectorList self, _VectorList& impl)

cdef class SharedVectorList(VectorList):
    cdef _shared.SharedItem[_VectorList]* shared_impl
    cdef SharedVectorList ShInit(SharedVectorList self, _shared.SharedItem[_VectorList]* impl)
    cpdef void H2D(SharedVectorList self)
    cpdef void D2H(SharedVectorList self)
