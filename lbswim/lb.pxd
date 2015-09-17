cimport numpy as np
# Get the c headers
cimport _lb

cimport shared

cdef class ArrayWrapper:
    cdef object owner
    cdef wrap(self, int ndim, np.npy_intp* shape, int dtype, void* data)

cdef class SharedLatticeArray:
    cdef shared.Array impl
    cdef np.ndarray view
    
cdef class LatticeAddressing:
    cdef _lb.LatticeAddressing* impl
    
cdef class LBParams:
    """Wraps all the parameters of the LB model.
    Treat this as read only.
    """
    cdef _lb.LBParams* impl
    cdef Lattice lat
    
cdef class Lattice:
    cdef _lb.Lattice* impl
    cdef LBParams _params
    cdef SharedLatticeArray _rho
    cdef SharedLatticeArray _u
    cdef SharedLatticeArray _force
    cdef SharedLatticeArray _fOld
    cdef SharedLatticeArray _fNew
    cdef LatticeAddressing _addr
