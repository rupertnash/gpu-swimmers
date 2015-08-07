cimport cython
cimport numpy as np
np.import_array()

# cdef void vectorise(shared.Array sa):
#     cdef np.PyArray_Dims newdims
#     newdims.len = 2
#     cdef np.npy_intp shape[2]
#     cdef int n = sa.impl.size
#     shape[0] = 3
#     shape[1] = n / 3
#     newdims.ptr = shape
#     np.PyArray_Resize(sa.view, cython.address(newdims), 1, np.NPY_CORDER)
    
cdef class Array:
    def __cinit__(self, int n, double hydro):
        self.impl = new SwimmerArray(n, hydro)
        self._r = shared.Array().init(self, cython.address(self.impl.r))
        self._r.data.shape = (3, n)
        
        self._v = shared.Array().init(self, cython.address(self.impl.v))
        self._v.data.shape = (3, n)
        
        self._n = shared.Array().init(self, cython.address(self.impl.n))
        self._n.data.shape = (3, n)
        
        self._P = shared.Array().init(self, cython.address(self.impl.P))
        self._a = shared.Array().init(self, cython.address(self.impl.a))
        self._l = shared.Array().init(self, cython.address(self.impl.l))
    
    cpdef AddForces(self, Lattice lat):
        self.impl.AddForces(lat.impl)

    cpdef Move(self, Lattice lat):
        self.impl.Move(lat.impl)

    def H2D(self):
        self._r.H2D()
        self._v.H2D()
        self._n.H2D()
        self._P.H2D()
        self._a.H2D()
        self._l.H2D()
        
    def D2H(self):
        self._r.D2H()
        self._v.D2H()
        self._n.D2H()
        self._P.D2H()
        self._a.D2H()
        self._l.D2H()

    @property
    def r(self):
        return self._r
    @property
    def v(self):
        return self._v
    @property
    def n(self):
        return self._n
    @property
    def P(self):
        return self._P
    @property
    def a(self):
        return self._a
    @property
    def l(self):
        return self._l

class System(object):
    def __init__(self, lat, sw):
        self.lat = lat
        self.sw = sw
        
    def Step(self):
        self.lat.ZeroForce()
        self.sw.AddForces(self.lat)
        self.lat.Step()
        self.lat.CalcHydro()
        self.sw.Move(self.lat)
    pass
