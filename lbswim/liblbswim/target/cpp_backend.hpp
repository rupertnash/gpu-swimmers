#ifndef TARGET_CPP_BACKEND_HPP
#define TARGET_CPP_BACKEND_HPP

#include "./cpp_backend.h"

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
  __target__ auto CppThreadContext<ND, VL>::operator[](size_t ilp_idx) const -> Shape {
    Shape ans = idx;
    ans[ND-1] += ilp_idx;
    return ans;
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

  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  __targetBoth__ Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::ArgTypes(const Index& shape) {
    static_assert(ND > 0, "Must have at least one dimension");
  }
  
  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  void Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::Launch(Args&&... args) {
    indexSpace = new CppContext<ND,VL>(extent);
    static_cast<Impl*>(this)->Run(std::forward<Args>(args)...);
    delete indexSpace;
  }
  
}
#endif
