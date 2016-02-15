// -*- mode: C++; -*-
#ifndef TARGET_VANILLA_BACKEND_H
#define TARGET_VANILLA_BACKEND_H

#include "./func_attr.h"
#include "../array.h"
#include "./function_traits.h"

namespace target {
  
  // Forward declarations
  template<size_t, size_t>
  struct CppThreadContext;
  template<size_t, size_t>
  struct CppSimdContext;

  // Target-only class that controls the iteration over an index space
  template<size_t ND = 1, size_t VL = VVL>
  struct CppContext {
  
    typedef array<size_t, ND> Shape;
  
    // Constructors
    __target__ CppContext(const Shape& n);
    __target__ CppContext(const size_t n);
  
    // Container protocol
    __target__ CppThreadContext<ND, VL> begin() const;
    __target__ CppThreadContext<ND, VL> end() const;
  
    // Describe the index space the global iteration is over
    const Shape start;
    const Shape extent;
  };

  // Target-only class that controls thread-level iteration over the
  // index space
  template<size_t ND = 1, size_t VL = VVL>
  struct CppThreadContext {
  
    typedef CppContext<ND, VL> Parent;
    typedef typename Parent::Shape Shape;
  
    // Constructor
    __target__ CppThreadContext(const Parent& ctx_, const Shape& pos);
  
    // Iterator protocol - note it returns itself on derefence!
    __target__ CppThreadContext& operator++();
    __target__ bool operator!=(const CppThreadContext& other) const;
    __target__ const CppThreadContext& operator*() const;

    __target__ Shape operator[](size_t ilp_idx) const;
    
    // Container protocol
    __target__ CppSimdContext<ND, VL> begin() const;
    __target__ CppSimdContext<ND, VL> end() const;
  
    // Global iteration space
    const Parent& ctx;
    // Current position of this thread in its iteration
    Shape idx;
  };
  
  // Target-only class for instruction-level iteration over the space
  template<size_t ND = 1, size_t VL = VVL>
  struct CppSimdContext {
  
    typedef CppThreadContext<ND, VL> Parent;
    typedef typename Parent::Shape Shape;
  
    // Constructor
    __target__ CppSimdContext(const Parent& ctx_, const size_t& pos);
  
    // Iterator protocol - derefernces to the current index
    __target__ CppSimdContext& operator++();
    __target__ bool operator!=(const CppSimdContext& other) const;
    __target__ Shape operator*() const;
  
    // Thread's iteration space
    const CppThreadContext<ND, VL>& ctx;
    // Current position
    size_t idx;
  };

  // Kernel Launcher - it knows the types it will be called with, but
  // you don't have to, as long as you use the factory function below
  template <size_t ND, class... FuncArgs>
  struct CppLauncher {

    typedef array<size_t, ND> ShapeT;  
    typedef void (*FuncT)(FuncArgs...);

    const ShapeT& shape;
    const FuncT& func;
  
    // Prepare to launch
    CppLauncher(const ShapeT& s, const FuncT f);
    // Launch!
    void operator()(FuncArgs... args);
  };

  // Factory function for global iteration contexts
  template <size_t ND, size_t VL>
  __target__ CppContext<ND, VL> MkContext(const array<size_t, ND>& shape);

  // Factory function for Launchers.
  //
  // Uses argument dependent lookup and function traits to construct
  // the right type of launcher
  template<class FuncT, class ShapeT>
  CppLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>
  MkLauncher(FuncT* f, ShapeT shape);

  template<class Impl>
  struct Kernel {
    template<size_t ND>
    struct Dims {
      typedef array<size_t, ND> Index;
      template<size_t VL>
      struct VecLen {
	template<class... Args>
	struct ArgTypes {
	  Index extent;
	  //dim3 nBlocks, blockShape;
	  CppContext<ND,VL>* indexSpace;
	  __targetBoth__ ArgTypes(const Index& shape);
	  void Launch(Args&&... args);
	  static const size_t nDim = ND;
	  static const size_t vecLen = VL;
	};
      };
    };
  };

}

#endif
