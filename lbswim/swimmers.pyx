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
            return self.impl.Host().P
        def __set__(self, double P):
            self.impl.Host().P = P
            self.impl.H2D()

    property l:
        def __get__(self):
            return self.impl.Host().l
        def __set__(self, double l):
            self.impl.Host().l = l
            self.impl.H2D()

    property alpha:
        def __get__(self):
            return self.impl.Host().alpha
        def __set__(self, double alpha):
            self.impl.Host().alpha = alpha
            self.impl.H2D()

    property mobility:
        def __get__(self):
            return self.impl.Host().mobility
        def __set__(self, double mobility):
            self.impl.Host().mobility = mobility
            self.impl.H2D()

    property translational_advection_off:
        def __get__(self):
            return self.impl.Host().translational_advection_off
        def __set__(self, bint translational_advection_off):
            self.impl.Host().translational_advection_off = translational_advection_off
            self.impl.H2D()

    property rotational_advection_off:
        def __get__(self):
            return self.impl.Host().rotational_advection_off
        def __set__(self, bint rotational_advection_off):
            self.impl.Host().rotational_advection_off = rotational_advection_off
            self.impl.H2D()

    property seed:
        def __get__(self):
            return self.impl.Host().seed
        def __set__(self, unsigned long long seed):
            self.impl.Host().seed = seed
            self.impl.H2D()
    
    def __reduce__(self):
        params = {}
        cdef SwimmerArray.CommonParams* tmp = self.impl.Host()
        params['P'] = tmp.P 
        params['l'] = tmp.l
        params['alpha'] = tmp.alpha
        params['mobility'] = tmp.mobility
        params['translational_advection_off'] = tmp.translational_advection_off
        params['rotational_advection_off'] = tmp.rotational_advection_off
        params['seed'] = tmp.seed
        
        return (CommonParams,
                (),
                params)
    
    def __setstate__(self, params):
        cdef SwimmerArray.CommonParams* tmp = self.impl.Host()
        tmp.P = params['P']
        tmp.l = params['l']
        tmp.alpha = params['alpha']
        tmp.mobility = params['mobility']
        tmp.translational_advection_off = params['translational_advection_off']
        tmp.rotational_advection_off = params['rotational_advection_off']
        tmp.seed = params['seed']
        self.impl.H2D()
        
cdef class Array:
    def __cinit__(self, int n, CommonParams cp):
        self.impl = new SwimmerArray.SwimmerArray(n, cp.impl.Host())

        self.r = SharedVectorList().ShInit(cython.address(self.impl.r))
        self.v = SharedVectorList().ShInit(cython.address(self.impl.v))
        self.n = SharedVectorList().ShInit(cython.address(self.impl.n))

        self.common = CommonParams()
        self.common.adopt(cython.address(self.impl.common))
        return
    
    cpdef AddForces(self, Lattice lat):
        self.impl.AddForces(lat.impl)

    cpdef Move(self, Lattice lat):
        self.impl.Move(lat.impl)

    def H2D(self):
        self.r.H2D()
        self.v.H2D()
        self.n.H2D()
        self.impl.prng.H2D()
        
    def D2H(self):
        self.r.D2H()
        self.v.D2H()
        self.n.D2H()
        self.impl.prng.D2H()
        
    def __reduce__(self):
        self.D2H()
        array_data = (self.r.data, self.v.data, self.n.data)
        return (Array,
                (self.impl.num, self.common),
                array_data)
    
    def __setstate__(self, data):
        r, v, n = data
        self.r.data[:] = r[:]
        self.v.data[:] = v[:]
        self.n.data[:] = n[:]
        self.H2D()
