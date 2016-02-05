from cpython cimport Py_buffer

cdef class Array:
    def __cinit__(self):
        self.data = None
    def __getattr__(self, attr):
        return getattr(self.data, attr)
    def __getitem__(self, idx):
        return self.data[idx]
    def __setitem__(self, idx, val):
        self.data[idx] = val
    
