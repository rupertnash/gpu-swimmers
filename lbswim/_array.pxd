cdef extern from "Fields.h":
    cdef cppclass ScalarField:
        pass
    cdef cppclass VectorField:
        pass
    cdef cppclass DistField:
        pass

cdef extern from "Lists.h":
    cdef cppclass ScalarList:
        pass
    cdef cppclass VectorList:
        pass
    cdef cppclass RandList:
        pass
