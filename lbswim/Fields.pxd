from dq cimport *
cimport _array
cimport Array
cimport _shared
cimport numpy as np

ctypedef _array.Array[double, DQ_d, ONE] _ScalarField
ctypedef _array.Array[double, DQ_d, DQ_d] _VectorField
ctypedef _array.Array[double, DQ_d, DQ_q] _DistField

cdef class ScalarField(Array.Array):
    cdef _array.ArrayHelper[_ScalarField]* impl
    cdef np.ndarray data
    cdef void Init(ScalarField self, _ScalarField* impl)

cdef class VectorField(Array.Array):
    cdef _array.ArrayHelper[_VectorField]* impl
    cdef np.ndarray data
    cdef void Init(VectorField self, _VectorField* impl)

cdef class DistField(Array.Array):
    cdef _array.ArrayHelper[_DistField]* impl
    cdef np.ndarray data
    cdef void Init(DistField self, _DistField* impl)

cdef class SharedScalarField(ScalarField):
    cdef _shared.SharedItem[_ScalarField]* shared_impl
    cdef void ShInit(SharedScalarField self, _shared.SharedItem[_ScalarField]* impl)
    cdef void H2D(SharedScalarField self)
    cdef void D2H(SharedScalarField self)

cdef class SharedVectorField(VectorField):
    cdef _shared.SharedItem[_VectorField]* shared_impl
    cdef void ShInit(SharedVectorField self, _shared.SharedItem[_VectorField]* impl)
    cdef void H2D(SharedVectorField self)
    cdef void D2H(SharedVectorField self)

cdef class SharedDistField(DistField):
    cdef _shared.SharedItem[_DistField]* shared_impl
    cdef void ShInit(SharedDistField self, _shared.SharedItem[_DistField]* impl)
    cdef void H2D(SharedDistField self)
    cdef void D2H(SharedDistField self)
