#from _array cimport ScalarField, VectorField, DistField, ScalarList, VectorList

cdef extern from "SharedArray.h":    
    cdef cppclass SharedItem[T]:
        void H2D()
        void D2H()
