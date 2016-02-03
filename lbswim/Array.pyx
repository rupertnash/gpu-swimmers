from cpython cimport Py_buffer

cdef class ScalarField:
    def __cinit__(self):
        self.impl = NULL
    cdef void Init(ScalarField self, _array.ScalarField* impl):
        self.impl = new _array.ArrayHelper[_array.ScalarField](impl)
        self.view = np.asarray(self)
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)
        

cdef class VectorField:
    def __cinit__(self):
        self.impl = NULL
    cdef void Init(VectorField self, _array.VectorField* impl):
        self.impl = new _array.ArrayHelper[_array.VectorField](impl)
        self.view = np.asarray(self)
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)

cdef class DistField:
    def __cinit__(self):
        self.impl = NULL
    cdef void Init(DistField self, _array.DistField* impl):
        self.impl = new _array.ArrayHelper[_array.DistField](impl)
        self.view = np.asarray(self)
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self.impl.GetBuffer(buffer, flags)
    def __releasebuffer__(self, Py_buffer* buffer):
        self.impl.ReleaseBuffer(buffer)
