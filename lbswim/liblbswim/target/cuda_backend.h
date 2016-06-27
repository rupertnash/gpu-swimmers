// -*- mode: C++; -*-
#ifndef TARGET_CUDA_BACKEND_H
#define TARGET_CUDA_BACKEND_H

#include "./func_attr.h"
#include "./array.h"
#include "./function_traits.h"
#include "./Nd.h"
#include <assert.h>

namespace target {
  
  // Forward declarations
  template<size_t, size_t>
  struct CudaThreadContext;
  template<class, size_t>
  struct VectorView;
  // template<size_t, size_t>
  // struct CudaSimdContext;

  // Target-only class that controls the iteration over an index space
  
  // Each CUDA thread is responsible for VL consecutive elements in
  // the flattened one-dimensional array. This could easily be
  // extended to grid-striding which might help.
  // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  template<size_t ND = 1, size_t VL = TARGET_DEFAULT_VVL>
  struct CudaContext {
  
    typedef array<size_t, ND> Shape;
  
    // Constructors
    __target__ CudaContext(const Shape& n);
    __target__ CudaContext(const size_t n);
  
    // Container protocol
    __target__ CudaThreadContext<ND, VL> begin() const;
    __target__ CudaThreadContext<ND, VL> end() const;

    // Describe the index space the global iteration is over
    size_t start, finish;
    const Shape extent;
    const SpaceIndexer<ND> indexer;
  };

  // Target-only class that controls thread-level iteration over the
  // index space
  template<size_t ND = 1, size_t VL = TARGET_DEFAULT_VVL>
  struct CudaThreadContext {
  
    typedef CudaContext<ND, VL> Parent;
    typedef typename Parent::Shape Shape;
  
    // Constructor
    __target__ CudaThreadContext(const Parent& ctx_, const size_t& pos);
  
    // Iterator protocol - note it returns itself on derefence!
    __target__ CudaThreadContext& operator++();
    __target__ bool operator!=(const CudaThreadContext& other) const;
    __target__ const CudaThreadContext& operator*() const;
  
    __target__ Shape operator[](size_t ilp_idx) const;

    // Get the current index slice from an array
    template <class ArrayT>
    __target__ VectorView<ArrayT, VL> GetCurrentElements(ArrayT* arr);
    
    __target__ Shape GetNdIndex(size_t i) const;
    
    __target__ constexpr size_t VectorLength() const {
      return VL;
    }
    
    // Global iteration space
    const Parent& ctx;
    // Current position of this thread in its iteration
    size_t ijk;
  };

  template<size_t ND, size_t VL>
  __target__ bool operator<(const CudaThreadContext<ND, VL>& a, const CudaThreadContext<ND, VL>& b);

  // Target-only class for instruction-level iteration over the space
  template<class AT, size_t VL = TARGET_DEFAULT_VVL>
  struct VectorView {
    typedef AT ArrayT;
    typedef typename AT::ElemType ElemType;
    typedef typename AT::WrapType WrapType;

    WrapType zero;
    __target__ WrapType operator[](size_t i);
    __target__ ElemType& operator()(size_t i, size_t d);
    __target__ const ElemType& operator()(size_t i, size_t d) const;
  };
  
 template<class Impl, size_t ND, size_t VL>
 struct Kernel {
   typedef array<size_t, ND> Index;
   Index extent;
   size_t nBlocks, blockShape;
   CudaContext<ND,VL>* indexSpace;
   __targetBoth__ Kernel(const Index& shape);
   template<class... Args>
   void operator()(Args... args);
   __targetBoth__ constexpr size_t Dims() const {
     return ND;
   }
   __targetBoth__ constexpr size_t VecLen() const {
     return VL;
   }
 };

}
#endif
