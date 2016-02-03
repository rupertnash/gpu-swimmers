from dq cimport *
cimport _array
cimport Array
cimport _shared
cimport numpy as np

ctypedef _array.Array[double, ONE, ONE] _ScalarList
ctypedef _array.Array[double, ONE, DQ_d] _VectorList

cdef class ScalarList(Array.Array):
    cdef _array.ArrayHelper[_ScalarList]* impl
    cdef np.ndarray data
    cdef void Init(ScalarList self, _ScalarList* impl)

cdef class SharedScalarList(ScalarList):
    cdef _shared.SharedItem[_ScalarList]* shared_impl
    cdef void ShInit(SharedScalarList self, _shared.SharedItem[_ScalarList]* impl)
    cdef void H2D(SharedScalarList self)
    cdef void D2H(SharedScalarList self)

cdef class VectorList(Array.Array):
    cdef _array.ArrayHelper[_VectorList]* impl
    cdef np.ndarray data
    cdef void Init(VectorList self, _VectorList* impl)

cdef class SharedVectorList(VectorList):
    cdef _shared.SharedItem[_VectorList]* shared_impl
    cdef void ShInit(SharedVectorList self, _shared.SharedItem[_VectorList]* impl)
    cdef void H2D(SharedVectorList self)
    cdef void D2H(SharedVectorList self)
