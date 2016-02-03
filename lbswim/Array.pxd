from dq cimport *
cimport _array
cimport numpy as np

cdef class ScalarField:
    cdef _array.ArrayHelper[_array.ScalarField]* impl
    cdef np.ndarray view
    cdef void Init(ScalarField self, _array.ScalarField* impl)

cdef class VectorField:
    cdef _array.ArrayHelper[_array.VectorField]* impl
    cdef np.ndarray view
    cdef void Init(VectorField self, _array.VectorField* impl)

cdef class DistField:
    cdef _array.ArrayHelper[_array.DistField]* impl
    cdef np.ndarray view
    cdef void Init(DistField self, _array.DistField* impl)
