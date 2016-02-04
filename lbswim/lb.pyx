cimport cython
np.import_array()

cdef class ArrayWrapper:
    def __cinit__(self, owner):
        self.owner = owner
        return
    cdef wrap(self, int ndim, np.npy_intp* shape, int dtype, void* data):
        ndarray = np.PyArray_SimpleNewFromData(ndim, shape,
                                                dtype, data)
        np.set_array_base(ndarray, self)
        return ndarray

cdef class LBParams:
    """Wraps all the parameters of the LB model.
    Treat this as read only.
    """
    @property
    def cs2(self):
        """Speed of sound, squared"""
        return self.impl.cs2
    @property
    def w(self):
        """Quadrature weights"""
        cdef np.npy_intp shape[1]
        shape[0] = _lb.DQ_q
        return ArrayWrapper(self).wrap(1, shape, np.NPY_DOUBLE, self.impl.w)
    @property
    def xi(self):
        """velocities - as doubles"""
        cdef np.npy_intp shape[2]
        shape[0] = _lb.DQ_q
        shape[1] = _lb.DQ_d
        return ArrayWrapper(self).wrap(2, shape, np.NPY_DOUBLE, self.impl.xi)
    @property
    def ci(self):
        """velocities - as ints"""
        cdef np.npy_intp shape[2]
        shape[0] = _lb.DQ_q
        shape[1] = _lb.DQ_d
        return ArrayWrapper(self).wrap(2, shape, np.NPY_INT, self.impl.ci)
    @property
    def Q(self):
        """Kinetic projector
        Q_i_ab = xi_i_a * xi_i_b - delta_ab cs^2
        """
        cdef np.npy_intp shape[3]
        shape[0] = _lb.DQ_q
        shape[1] = _lb.DQ_d
        shape[2] = _lb.DQ_d
        return ArrayWrapper(self).wrap(3, shape, np.NPY_DOUBLE, self.impl.Q)
    @property
    def complement(self):
        cdef np.npy_intp shape[1]
        shape[0] = _lb.DQ_q
        return ArrayWrapper(self).wrap(1, shape, np.NPY_INT, self.impl.complement)
    @property
    def norms(self):
        """The modes' normalizers"""
        cdef np.npy_intp shape[1]
        shape[0] = _lb.DQ_q
        return ArrayWrapper(self).wrap(1, shape, np.NPY_DOUBLE, self.impl.norms)
    @property
    def mm(self):
        """Matrix of eigenvectors (they are the rows)"""
        cdef np.npy_intp shape[2]
        shape[0] = _lb.DQ_q
        shape[1] = _lb.DQ_q
        return ArrayWrapper(self).wrap(2, shape, np.NPY_DOUBLE, self.impl.mm)
    
    @property
    def mmi(self):
        """Inverse matrix"""
        cdef np.npy_intp shape[2]
        shape[0] = _lb.DQ_q
        shape[1] = _lb.DQ_q
        return ArrayWrapper(self).wrap(2, shape, np.NPY_DOUBLE, self.impl.mmi)

    @property
    def tau_s(self):
        """Shear relaxation time.
        """
        return self.impl.tau_s
    
    @property
    def eta(self):
        """Viscosity of the fluid"""
        return self.tau_s * self.cs2
    
    @property
    def tau_b(self):
        """Bulk relaxation time.
        """
        return self.impl.tau_b
    
cdef class Lattice:
    def __cinit__(self, int nx, int ny, int nz, double tau_s, double tau_b):
        cdef _lb.Shape shp = _lb.Shape(nx, ny, nz)
        self.impl = new _lb.Lattice(shp, tau_s, tau_b)
        if self.impl is NULL:
            raise MemoryError()
        
        self._params = None
        self.rho = SharedScalarField().ShInit(cython.address(self.impl.data.rho))
        self.u = SharedVectorField().ShInit(cython.address(self.impl.data.u))
        self.force = SharedVectorField().ShInit(cython.address(self.impl.data.force))
        self.fOld = SharedDistField().ShInit(cython.address(self.impl.data.fOld))
        self.fNew = SharedDistField().ShInit(cython.address(self.impl.data.fNew))
    
    def __dealloc__(self):
        if self.impl is not NULL:
            del self.impl
            
    def Step(self):
        self.impl.Step()
        
    def CalcHydro(self):
        self.impl.CalcHydro()
        
    def InitFromHydro(self):
        self.impl.InitFromHydro()
    
    def ZeroForce(self):
        self.impl.ZeroForce()

    @property
    def time_step(self):
        return self.impl.time_step
        
    @property
    def params(self):
        if self._params is None:
            self._params = LBParams()
            self._params.impl = self.impl.params.Host()
            self._params.lat = self
        return self._params
    
    def __reduce__(self):
        """Implement pickle protocol for extension classes.
        Return 3-tuple of (constructor, ctor_args, setstate_args)
        """
        raise NotImplementedError()
        # cdef int* size = self.impl.addr.host.size
        # cdef _lb.LBParams* p = self.impl.params.host
        # init_args = (size[0], size[1], size[2], p.tau_s, p.tau_b)

        # # fOld is the only array that really matters, the rest can be recomputed
        # self.fOld.D2H()
        # data = (self.impl.time_step, self.fOld.data)
        
        # return (self.__class__,
        #         init_args,
        #         data)

    def __setstate__(self, data):
        # ts, file_f_data = data
        
        # self.impl.time_step = ts
        
        # my_f_data = self.fOld.data
        # my_f_data[:] = file_f_data[:]
        # self.fOld.H2D()
        raise NotImplementedError()
