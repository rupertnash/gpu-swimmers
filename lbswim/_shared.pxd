cdef extern from "Shared.h":
    cdef cppclass SharedArray[T]:
        T* host
        T* device
        size_t size
        void H2D()
        void D2H()
    
    cdef cppclass SharedItem[T]:
        T* host
        T* device
        void H2D()
        void D2H()
