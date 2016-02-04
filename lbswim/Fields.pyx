from cpython cimport Py_buffer
np.import_array()

cdef class ScalarField:
    def __cinit__(self):
        self.impl = NULL
    cdef ScalarField Init(ScalarField self, _ScalarField* impl):
        self.impl = new _array.ArrayHelper[_ScalarField](impl)
        self.data = np.PyArray_FROM_O(self)
        return self
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)
        

cdef class VectorField:
    def __cinit__(self):
        self.impl = NULL
    cdef VectorField Init(VectorField self, _VectorField* impl):
        self.impl = new _array.ArrayHelper[_VectorField](impl)
        self.data = np.PyArray_FROM_O(self)
        return self
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)

cdef class DistField:
    def __cinit__(self):
        self.impl = NULL
    cdef DistField Init(DistField self, _DistField* impl):
        self.impl = new _array.ArrayHelper[_DistField](impl)
        self.data = np.PyArray_FROM_O(self)
        return self
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)

cdef class SharedScalarField:
    def __cinit__(self):
        self.shared_impl = NULL
    
    cdef SharedScalarField ShInit(SharedScalarField self, _shared.SharedItem[_ScalarField]* impl):
        self.shared_impl = impl
        ScalarField.Init(self, impl.Host())
        return self
    cpdef void H2D(SharedScalarField self):
        self.shared_impl.H2D()
    cpdef void D2H(SharedScalarField self):
        self.shared_impl.D2H()

cdef class SharedVectorField:
    def __cinit__(self):
        self.shared_impl = NULL
    
    cdef SharedVectorField ShInit(SharedVectorField self, _shared.SharedItem[_VectorField]* impl):
        self.shared_impl = impl
        VectorField.Init(self, impl.Host())
        return self
    
    cdef void H2D(SharedVectorField self):
        self.shared_impl.H2D()
    cdef void D2H(SharedVectorField self):
        self.shared_impl.D2H()

cdef class SharedDistField:
    def __cinit__(self):
        self.shared_impl = NULL
    
    cdef SharedDistField ShInit(SharedDistField self, _shared.SharedItem[_DistField]* impl):
        self.shared_impl = impl
        DistField.Init(self, impl.Host())
        return self
    
    cdef void H2D(SharedDistField self):
        self.shared_impl.H2D()
    cdef void D2H(SharedDistField self):
        self.shared_impl.D2H()
