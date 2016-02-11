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
}

namespace target {
  template<size_t ND, size_t VL>
  __target__ CppContext<ND,VL>::CppContext(const Shape& n) : start(), extent(n) {
    static_assert(ND > 0, "Must have at least one dimension");
  }

  template<size_t ND, size_t VL>
  __target__ CppContext<ND,VL>::CppContext(const size_t n) : CppContext(Shape(n)) {
    static_assert(ND == 1, "Context construction from integer only allowed in 1D");
  }
  
  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL> CppContext<ND,VL>::begin() const {
    return CppThreadContext<ND, VL>(*this, start);
  }
  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL> CppContext<ND,VL>::end() const {
    return CppThreadContext<ND, VL>(*this, extent);
  }

  // Now CppThreadContext
  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL>::CppThreadContext(const Parent& ctx_, const Shape& pos) : ctx(ctx_), idx(pos) {
  }
  
  template<size_t ND, size_t VL, size_t iDim>
  struct inc {
    static bool helper(array<size_t, ND>& i, const array<size_t,ND>& n) {
      if (iDim == (ND-1))
	i[iDim] += VL;//std::min(i[iDim] + VL, n[iDim]);
      else
	i[iDim]++;
      
      if (i[iDim] == n[iDim]) {
	if (iDim == 0)
	  return true;
	// roll over next index
	bool done = inc<ND, VL, iDim - 1>::helper(i, n);
	
	if (done) return true;
	
	i[iDim] = 0;
      }
      return false;
    }
  };
  template<size_t ND, size_t VL>
  struct inc<ND, VL, 0> {
    static bool helper(array<size_t, ND>& i, const array<size_t,ND>& n) {
      if (ND == 1)
	i[0] += VL; //std::min(i[0] + VL, n[0]);
      else
	i[0]++;
      
      if (i[0] == n[0]) {
	return true;
      }
      return false;
    }
  };

  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL>& CppThreadContext<ND, VL>::operator++() {
    inc<ND, VL, ND-1>::helper(idx, ctx.extent);
    return *this;
  }
  
  template<size_t ND, size_t VL>  
  __target__ bool CppThreadContext<ND, VL>::operator!=(const CppThreadContext& other) const {
    if (&ctx != &other.ctx)
      return true;  
    if (idx != other.idx)
      return true;
    return false;
  }
  
  template<size_t ND, size_t VL>
  __target__ const CppThreadContext<ND, VL>& CppThreadContext<ND, VL>::operator*() const {
    return *this;
  }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL> CppThreadContext<ND, VL>::begin() const {
    return CppSimdContext<ND, VL>(*this, 0);
  }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL> CppThreadContext<ND, VL>::end() const {
    return CppSimdContext<ND, VL>(*this, VL);
  }

  // CppSimdContext
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL>::CppSimdContext(const Parent& ctx_, const size_t& pos)
    : ctx(ctx_), idx(pos) {
  }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL>& CppSimdContext<ND, VL>::operator++() {
    ++idx;
    return *this;
  }

  template<size_t ND, size_t VL>
  __target__ bool CppSimdContext<ND, VL>::operator!=(const CppSimdContext& other) const {
    if (&ctx != &other.ctx)
      return true;
    if (idx != other.idx)
      return true;
    return false;
  }

  template<size_t ND, size_t VL>
  __target__ auto CppSimdContext<ND, VL>::operator*() const -> Shape {
    Shape ans(ctx.idx);
    ans[ND-1] += idx;
    return ans;
  }

  // Factory function for global iteration context
  template <size_t ND>
  __target__ CppContext<ND> MkContext(const array<size_t, ND>& shape) {
    return CppContext<ND>(shape);
  }


  // Kernel Launcher - knows the types it will be called with.
  template <size_t ND, class... FuncArgs>
  CppLauncher<ND, FuncArgs...>::CppLauncher(const ShapeT& s, const FuncT f) : shape(s), func(f) {
    static_assert(ND > 0, "Must have at least one dimension");
  }
  
  template <size_t ND, class... FuncArgs>
  void CppLauncher<ND, FuncArgs...>::operator()(FuncArgs... args) {
    func(args...);
  }

  // Partial specialisation on the pseudo typedef of the args
  // parameter pack. This subclasses the class we actually want but
  // then inherits its constructor.
  template <size_t ND, class... FuncArgs>
  struct CppLauncher<ND, variadic_typedef<FuncArgs...> > : public CppLauncher<ND, FuncArgs...>
  {
    // Inherit c'tor
    using CppLauncher<ND, FuncArgs...>::CppLauncher;
  };
  // Factory function for Launchers.
  //
  // Uses the function traits and the specialisation on the variadic
  // pseudo typedef to construct the right type of launcher
  template<class FuncT, class ShapeT>
  CppLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>
  launch(FuncT* f, ShapeT shape) {
    return CppLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>(shape, f);
  }


}
#endif
