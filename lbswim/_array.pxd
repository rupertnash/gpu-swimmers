from cpython cimport Py_buffer
from dq cimport *

cdef extern from "array.h":
    cdef cppclass array[T, N]:
        pass

cdef extern from "Array.h":
    cdef cppclass SpaceIndexer[NDIM]:
        array[size_t, NDIM] shape
        array[size_t, NDIM] strides
        size_t size
  
    cdef cppclass Array[T, NDIM, NELEM]:
        SpaceIndexer[NDIM] indexer
        T* data
        T* baseData
        size_t nElems()
        size_t nDims()

ctypedef Array[double, DQ_d, ONE] ScalarField
ctypedef Array[double, DQ_d, DQ_d] VectorField
ctypedef Array[double, DQ_d, DQ_q] DistField

ctypedef Array[double, ONE, ONE] ScalarList
ctypedef Array[double, ONE, DQ_d] VectorList

cdef extern from "ArrayHelper.h":
    cdef cppclass ArrayHelper[ArrayType]:
        ArrayHelper(ArrayType* impl_)
        void GetBuffer(Py_buffer* view, int flags)
        void ReleaseBuffer(Py_buffer* view)
