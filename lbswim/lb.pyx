cimport cython
# Import numpy both ways
import numpy as np
#cimport numpy as np
np.import_array()
# Get the c headers
#cimport _lb

#cimport shared

cdef class ArrayWrapper:
    def __cinit__(self, owner):
        self.owner = owner
        return
    cdef wrap(self, int ndim, np.npy_intp* shape, int dtype, void* data):
        ndarray = np.PyArray_SimpleNewFromData(ndim, shape,
                                                dtype, data)
        np.set_array_base(ndarray, self)
        return ndarray

cdef class SharedLatticeArray:
    def __init__(self, Lattice lat, shared.Array sa):
        self.impl = sa
        cdef int nSites = lat.addr.n
        shape = (sa.impl.size/nSites, )+ lat.addr.size
        self.view = sa.view.reshape(shape)
        return
    
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
    pass

cdef class LatticeAddressing:
    @property
    def size(self):
        return tuple(self.impl.size)
    
    @property
    def n(self):
        return self.impl.n
    
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
        self.impl = new _lb.Lattice(nx, ny, nz, tau_s, tau_b)
        self._params = None
        self._rho = None
        self._u = None
        self._force = None
        self._fOld = None
        self._fNew = None
        if self.impl is NULL:
            raise MemoryError()
    
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
    def addr(self):
        ans = LatticeAddressing()
        ans.impl = self.impl.addr.host
        return ans
    
    @property
    def params(self):
        if self._params is None:
            self._params = LBParams()
            self._params.impl = self.impl.params.host
            self._params.lat = self
        return self._params
    
    @property
    def rho(self):
        if self._rho is None:
            raw = shared.Array().init(self, cython.address(self.impl.data.rho))
            self._rho = SharedLatticeArray(self, raw)
        return self._rho

    @property
    def u(self):
        if self._u is None:
            raw = shared.Array().init(self, cython.address(self.impl.data.u))
            self._u = SharedLatticeArray(self, raw)
        return self._u

    @property
    def force(self):
        if self._force is None:
            raw = shared.Array().init(self, cython.address(self.impl.data.force))
            self._force = SharedLatticeArray(self, raw)
        return self._force

    @property
    def fOld(self):
        if self._fOld is None:
            raw = shared.Array().init(self, cython.address(self.impl.data.fOld))
            self._fOld = SharedLatticeArray(self, raw)
        return self._fOld

    @property
    def fNew(self):
        if self._fNew is None:
            raw = shared.Array().init(self, cython.address(self.impl.data.fNew))
            self._fNew = SharedLatticeArray(self, raw)
        return self._fNew
