cdef extern from "SharedNdArray.h":
    cdef cppclass SharedItem[T]:
        void H2D()
        void D2H()
        T& Host()
        T& Device()
