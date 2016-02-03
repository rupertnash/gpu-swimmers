cimport cython
cimport numpy as np
np.import_array()

cdef class Array:
    def __cinit__(self, int n):
        self.impl = new TracerArray.TracerArray(n)
        
        self.r.ShInit(cython.address(self.impl.r))
        self.s.ShInit(cython.address(self.impl.s))
        self.v.ShInit(cython.address(self.impl.v))
        
        return
    
    cpdef AddForces(self, Lattice lat):
        pass

    cpdef Move(self, Lattice lat):
        self.impl.Move(lat.impl)

    def H2D(self):
        self.r.H2D()
        self.s.H2D()
        self.v.H2D()
        
    def D2H(self):
        self.r.D2H()
        self.s.D2H()
        self.v.D2H()

    def __reduce__(self):
        self.D2H()
        array_data = (self.r.data, self.s.data, self.v.data)
        return (Array,
                (self.impl.num,),
                array_data)
    
    def __setstate__(self, data):
        r, s, v = data
        self.r.data[:] = r[:]
        self.s.data[:] = s[:]
        self.v.data[:] = v[:]
        self.H2D()
