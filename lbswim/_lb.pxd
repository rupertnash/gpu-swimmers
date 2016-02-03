from _shared cimport SharedItem
from Fields cimport _ScalarField, _VectorField, _DistField

cdef extern from "Lattice.h":
    enum: DQ_d
    enum: DQ_q

    cdef cppclass Shape:
        Shape()
        Shape(size_t, size_t, size_t)
        size_t operator[](size_t)
        
    cdef cppclass LBParams:
        double cs2
        double w[DQ_q]
        double xi[DQ_q][DQ_d]
        int ci[DQ_q][DQ_d]
        double Q[DQ_q][DQ_d][DQ_d]
        int complement[DQ_q]
        double norms[DQ_q]
        double mm[DQ_q][DQ_q]
        double mmi[DQ_q][DQ_q]
        double tau_s
        double tau_b
    
    cdef cppclass LDView:
        _ScalarField* rho
        _VectorField* u
        _VectorField* force
        _DistField* fOld
        _DistField* fNew

    cdef cppclass LatticeData:
        SharedItem[_ScalarField] rho
        SharedItem[_VectorField] u
        SharedItem[_VectorField] force
        SharedItem[_DistField] fOld
        SharedItem[_DistField] fNew
        
    cdef cppclass Lattice:
        Lattice(const Shape&, double, double)
        void Step()
        void CalcHydro()
        void InitFromHydro()
        void ZeroForce()
        SharedItem[LBParams] params
        Shape shape
        LatticeData data
        int time_step
    
