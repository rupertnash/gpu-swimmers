from cpython cimport Py_buffer
np.import_array()

cdef class ScalarList:
    def __cinit__(self):
        self.impl = NULL
    cdef ScalarList Init(ScalarList self, _ScalarList& impl):
        self.impl = new _array.ArrayHelper[_ScalarList](impl)
        self.data = np.PyArray_FROM_O(self)
        return self
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)

cdef class SharedScalarList:
    def __cinit__(self):
        self.shared_impl = NULL
    
    cdef SharedScalarList ShInit(SharedScalarList self, _shared.SharedItem[_ScalarList]* impl):
        self.shared_impl = impl
        ScalarList.Init(self, impl.Host())
        return self
    
    cpdef void H2D(SharedScalarList self):
        self.shared_impl.H2D()
    cpdef void D2H(SharedScalarList self):
        self.shared_impl.D2H()

cdef class VectorList:
    def __cinit__(self):
        self.impl = NULL
    cdef VectorList Init(VectorList self, _VectorList& impl):
        self.impl = new _array.ArrayHelper[_VectorList](impl)
        self.data = np.PyArray_FROM_O(self)
        return self
    
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)

cdef class SharedVectorList:
    def __cinit__(self):
        self.shared_impl = NULL
    
    cdef SharedVectorList ShInit(SharedVectorList self, _shared.SharedItem[_VectorList]* impl):
        self.shared_impl = impl
        VectorList.Init(self, impl.Host())
        return self
    
    cpdef void H2D(SharedVectorList self):
        self.shared_impl.H2D()
    cpdef void D2H(SharedVectorList self):
        self.shared_impl.D2H()
