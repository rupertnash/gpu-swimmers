cdef extern from "target/SharedNdArray.h" namespace "target":
    cdef cppclass SharedItem[T]:
        void H2D()
        void D2H()
        T& Host()
        T& Device()
