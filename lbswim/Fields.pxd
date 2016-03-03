from dq cimport *
cimport _array
cimport Array
cimport _shared
cimport numpy as np

ctypedef _array.NdArray[double, DQ_d, ONE] _ScalarField
ctypedef _array.NdArray[double, DQ_d, DQ_d] _VectorField
ctypedef _array.NdArray[double, DQ_d, DQ_q] _DistField

cdef class ScalarField(Array.Array):
    cdef _array.ArrayHelper[_ScalarField]* impl
    cdef ScalarField Init(ScalarField self, _ScalarField& impl)

cdef class VectorField(Array.Array):
    cdef _array.ArrayHelper[_VectorField]* impl
    cdef VectorField Init(VectorField self, _VectorField& impl)

cdef class DistField(Array.Array):
    cdef _array.ArrayHelper[_DistField]* impl
    cdef DistField Init(DistField self, _DistField& impl)

cdef class SharedScalarField(ScalarField):
    cdef _shared.SharedItem[_ScalarField]* shared_impl
    cdef SharedScalarField ShInit(SharedScalarField self, _shared.SharedItem[_ScalarField]* impl)
    cpdef void H2D(SharedScalarField self)
    cpdef void D2H(SharedScalarField self)

cdef class SharedVectorField(VectorField):
    cdef _shared.SharedItem[_VectorField]* shared_impl
    cdef SharedVectorField ShInit(SharedVectorField self, _shared.SharedItem[_VectorField]* impl)
    cpdef void H2D(SharedVectorField self)
    cpdef void D2H(SharedVectorField self)

cdef class SharedDistField(DistField):
    cdef _shared.SharedItem[_DistField]* shared_impl
    cdef SharedDistField ShInit(SharedDistField self, _shared.SharedItem[_DistField]* impl)
    cpdef void H2D(SharedDistField self)
    cpdef void D2H(SharedDistField self)
