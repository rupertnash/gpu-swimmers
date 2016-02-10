// -*- mode: C++; -*-
#ifndef TARGET_CUDA_BACKEND_H
#define TARGET_CUDA_BACKEND_H

#include "./func_attr.h"
#include "../array.h"
#include "./function_traits.h"

namespace target {
  
  // Forward declarations
  template<size_t, size_t>
  struct CudaThreadContext;
  template<size_t, size_t>
  struct CudaSimdContext;

  // Target-only class that controls the iteration over an index space
  template<size_t ND = 1, size_t VL = VVL>
  struct CudaContext {
  
    typedef array<size_t, ND> Shape;
  
    // Constructors
    __target__ CudaContext(const Shape& n);
    __target__ CudaContext(const size_t n);
  
    // Container protocol
    __target__ CudaThreadContext<ND, VL> begin() const;
    __target__ CudaThreadContext<ND, VL> end() const;
  
    // Describe the index space the global iteration is over
    const Shape extent;
    // The indices this CUDA thread will deal with
    Shape start, finish;
  };

  // Target-only class that controls thread-level iteration over the
  // index space
  template<size_t ND = 1, size_t VL = VVL>
  struct CudaThreadContext {
  
    typedef CudaContext<ND, VL> Parent;
    typedef typename Parent::Shape Shape;
  
    // Constructor
    __target__ CudaThreadContext(const Parent& ctx_, const Shape& pos);
  
    // Iterator protocol - note it returns itself on derefence!
    __target__ CudaThreadContext& operator++();
    __target__ bool operator!=(const CudaThreadContext& other) const;
    __target__ const CudaThreadContext& operator*() const;
  
    // Container protocol
    __target__ CudaSimdContext<ND, VL> begin() const;
    __target__ CudaSimdContext<ND, VL> end() const;
  
    // Global iteration space
    const Parent& ctx;
    // Current position of this thread in its iteration
    Shape idx;
  };

  // Target-only class for instruction-level iteration over the space
  template<size_t ND = 1, size_t VL = VVL>
  struct CudaSimdContext {
  
    typedef CudaThreadContext<ND, VL> Parent;
    typedef typename Parent::Shape Shape;
  
    // Constructor
    __target__ CudaSimdContext(const Parent& ctx_, const size_t& pos);
  
    // Iterator protocol - derefernces to the current index
    __target__ CudaSimdContext& operator++();
    __target__ bool operator!=(const CudaSimdContext& other) const;
    __target__ Shape operator*() const;
  
    // Thread's iteration space
    const CudaThreadContext<ND, VL>& ctx;
    // Current position
    size_t idx;
  };

  // Kernel Launcher - it knows the types it will be called with, but
  // you don't have to, as long as you use the factory function below
  template <size_t ND, class... FuncArgs>
  struct CudaLauncher {

    typedef array<size_t, ND> ShapeT;  
    typedef void (*FuncT)(FuncArgs...);

    const ShapeT& shape;
    const FuncT& func;
  
    dim3 nBlocks, blockShape;
  
    // Prepare to launch
    CudaLauncher(const ShapeT& s, const FuncT f);
    // Launch!
    void operator()(FuncArgs... args);
  };

  // Factory function for global iteration contexts
  template <size_t ND, size_t VL>
  __target__ CudaContext<ND, VL> MkContext(const array<size_t, ND>& shape);

  // Factory function for Launchers.
  //
  // Uses argument dependent lookup and function traits to construct
  // the right type of launcher
  template<class FuncT, class ShapeT>
  CudaLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>
  MkLauncher(FuncT* f, ShapeT shape);
}
#endif
