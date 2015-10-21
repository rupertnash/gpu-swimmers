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

    property l:
        def __get__(self):
            return self.impl.host.l
        def __set__(self, double l):
            self.impl.host.l = l
            self.impl.H2D()

    property alpha:
        def __get__(self):
            return self.impl.host.alpha
        def __set__(self, double alpha):
            self.impl.host.alpha = alpha
            self.impl.H2D()

    property mobility:
        def __get__(self):
            return self.impl.host.mobility
        def __set__(self, double mobility):
            self.impl.host.mobility = mobility
            self.impl.H2D()

    property translational_advection_off:
        def __get__(self):
            return self.impl.host.translational_advection_off
        def __set__(self, bint translational_advection_off):
            self.impl.host.translational_advection_off = translational_advection_off
            self.impl.H2D()

    property rotational_advection_off:
        def __get__(self):
            return self.impl.host.rotational_advection_off
        def __set__(self, bint rotational_advection_off):
            self.impl.host.rotational_advection_off = rotational_advection_off
            self.impl.H2D()

    property seed:
        def __get__(self):
            return self.impl.host.seed
        def __set__(self, unsigned long long seed):
            self.impl.host.seed = seed
            self.impl.H2D()
    
    def __reduce__(self):
        params = {}
        params['P'] = self.impl.host.P 
        params['l'] = self.impl.host.l
        params['alpha'] = self.impl.host.alpha
        params['mobility'] = self.impl.host.mobility
        params['translational_advection_off'] = self.impl.host.translational_advection_off
        params['rotational_advection_off'] = self.impl.host.rotational_advection_off
        params['seed'] = self.impl.host.seed
        
        return (CommonParams,
                (),
                params)
    
    def __setstate__(self, params):
        self.impl.host.P = params['P']
        self.impl.host.l = params['l']
        self.impl.host.alpha = params['alpha']
        self.impl.host.mobility = params['mobility']
        self.impl.host.translational_advection_off = params['translational_advection_off']
        self.impl.host.rotational_advection_off = params['rotational_advection_off']
        self.impl.host.seed = params['seed']
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
        self.impl.prng.H2D()
        
    def D2H(self):
        self._r.D2H()
        self._v.D2H()
        self._n.D2H()
        self.impl.prng.D2H()
    
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

    def __reduce__(self):
        self.D2H()
        array_data = (self._r.data, self._v.data, self._n.data)
        return (Array,
                (self.impl.num, self._common),
                array_data)
    
    def __setstate__(self, data):
        r, v, n = data
        self._r.data[:] = r[:]
        self._v.data[:] = v[:]
        self._n.data[:] = n[:]
        self.H2D()
