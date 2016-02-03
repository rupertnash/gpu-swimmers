cimport numpy as np
# Get the c headers
cimport _lb
cimport _array
from Fields cimport SharedScalarField, SharedVectorField, SharedDistField

cdef class ArrayWrapper:
    """Simple wrapper for C-style arrays into Numpy ndarrays.
    """
    cdef object owner
    cdef wrap(self, int ndim, np.npy_intp* shape, int dtype, void* data)
    
cdef class LBParams:
    """Wraps all the parameters of the LB model.
    Treat this as read only.
    """
    cdef _lb.LBParams* impl
    cdef Lattice lat
    
cdef class Lattice:
    cdef _lb.Lattice* impl
    cdef LBParams _params
    cdef SharedScalarField rho
    cdef SharedVectorField u
    cdef SharedVectorField force
    cdef SharedDistField fOld
    cdef SharedDistField fNew
