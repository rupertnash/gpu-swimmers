#cimport _shared
cimport cython
cimport numpy as np
np.import_array()

cdef class Array:
    cdef Array init(self, owner, _shared.SharedArray[double]* sa):
        self.owner = owner
        self.impl = sa
        cdef np.npy_intp size = self.impl.size
        cdef int ndim = 1
        self.view = np.PyArray_SimpleNewFromData(1,
                                                  cython.address(size),
                                                  np.NPY_DOUBLE,
                                                  self.impl.host)
        np.set_array_base(self.view, self)
        return self
    
    @property
    def data(self):
        return self.view
    
    def H2D(self):
        self.impl.H2D()
        return
    def D2H(self):
        self.impl.D2H()
        return
    
    def __getattr__(self, attr):
        return getattr(self.view, attr)
    def __getitem__(self, idx):
        return self.view[idx]
    def __setitem__(self, idx, val):
        self.view[idx] = val
