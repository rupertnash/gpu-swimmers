cdef extern from "SharedArray.h":    
    cdef cppclass SharedItem[T]:
        void H2D()
        void D2H()
        T* Host()
        T* Device()
