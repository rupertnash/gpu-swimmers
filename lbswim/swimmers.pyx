cimport cython
cimport numpy as np
np.import_array()

cdef class CommonParams:
    def __cinit__(self):
        self.impl = new _shared.SharedItem[SwimmerArray.CommonParams]()
        self.owner = True
        return
    
    cdef adopt(self, _shared.SharedItem[SwimmerArray.CommonParams]* cp):
        if self.owner:
            del self.impl
        self.impl = cp
        self.owner = False
        
    def __dealloc__(self):
        if self.owner:
            del self.impl

    property P:
        def __get__(self):
            return self.impl.host.P
        def __set__(self, double P):
            self.impl.host.P = P
            self.impl.H2D()

    property a:
        def __get__(self):
            return self.impl.host.a
        def __set__(self, double a):
            self.impl.host.a = a
            self.impl.H2D()

    property l:
        def __get__(self):
            return self.impl.host.l
        def __set__(self, double l):
            self.impl.host.l = l
            self.impl.H2D()

    property hydroRadius:
        def __get__(self):
            return self.impl.host.hydroRadius
        def __set__(self, double hydroRadius):
            self.impl.host.hydroRadius = hydroRadius
            self.impl.H2D()
    
    property alpha:
        def __get__(self):
            return self.impl.host.alpha
        def __set__(self, double alpha):
            self.impl.host.alpha = alpha
            self.impl.H2D()

    property seed:
        def __get__(self):
            return self.impl.host.seed
        def __set__(self, unsigned long long seed):
            self.impl.host.seed = seed
            self.impl.H2D()

cdef class Array:
    def __cinit__(self, int n, CommonParams cp):
        self.impl = new SwimmerArray.SwimmerArray(n, cp.impl.host)
        
        self._r = shared.Array().init(self, cython.address(self.impl.r))
        self._r.data.shape = (3, n)
        
        self._v = shared.Array().init(self, cython.address(self.impl.v))
        self._v.data.shape = (3, n)
        
        self._n = shared.Array().init(self, cython.address(self.impl.n))
        self._n.data.shape = (3, n)

        self._common = CommonParams()
        self._common.adopt(cython.address(self.impl.common))
        return
    
    cpdef AddForces(self, Lattice lat):
        self.impl.AddForces(lat.impl)

    cpdef Move(self, Lattice lat):
        self.impl.Move(lat.impl)

    def H2D(self):
        self._r.H2D()
        self._v.H2D()
        self._n.H2D()
        
    def D2H(self):
        self._r.D2H()
        self._v.D2H()
        self._n.D2H()

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
    def common(self):
        return self._common
