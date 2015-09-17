cimport cython
cimport numpy as np
np.import_array()

cdef class Array:
    def __cinit__(self, int n):
        self.impl = new TracerArray.TracerArray(n)
        
        self._r = shared.Array().init(self, cython.address(self.impl.r))
        self._r.data.shape = (3, n)
        
        self._v = shared.Array().init(self, cython.address(self.impl.v))
        self._v.data.shape = (3, n)
        
        return
    
    cpdef AddForces(self, Lattice lat):
        pass

    cpdef Move(self, Lattice lat):
        self.impl.Move(lat.impl)

    def H2D(self):
        self._r.H2D()
        self._v.H2D()
        
    def D2H(self):
        self._r.D2H()
        self._v.D2H()

    @property
    def r(self):
        return self._r
    @property
    def v(self):
        return self._v
    def __reduce__(self):
        self.D2H()
        array_data = (self._r.data, self._v.data)
        return (Array,
                (self.impl.num,),
                array_data)
    
    def __setstate__(self, data):
        r, v = data
        self._r.data[:] = r[:]
        self._v.data[:] = v[:]
        self.H2D()
