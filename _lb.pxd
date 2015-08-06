from _shared cimport SharedArray, SharedItem

cdef extern from "lbswim/lb.h":
    enum: DQ_d
    enum: DQ_q
    
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
        
    cdef cppclass LatticeAddressing:
        int size[DQ_d]
        int strides[DQ_d]
        int n

    # cdef cppclass LDView:
    #     double* rho
    #     double* u
    #     double* force
    #     double* fOld
    #     double* fNew

    cdef cppclass LatticeData:
        SharedArray[double] rho
        SharedArray[double] u
        SharedArray[double] force
        SharedArray[double] fOld
        SharedArray[double] fNew
        
	
    cdef cppclass Lattice:
        Lattice(int, int, int, double, double)
        void Step()
        void CalcHydro()
        void InitFromHydro()
        void ZeroForce()
        LatticeData* data
        SharedItem[LBParams] params
        SharedItem[LatticeAddressing] addr
        int time_step
    
