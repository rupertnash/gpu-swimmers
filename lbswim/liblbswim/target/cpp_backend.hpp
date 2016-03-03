#ifndef TARGET_CPP_BACKEND_HPP
#define TARGET_CPP_BACKEND_HPP

#include "./cpp_backend.h"

namespace target {
    
  template<size_t ND, size_t VL>
  __target__ CppContext<ND,VL>::CppContext(const Shape& n) : start(), extent(n), indexer(n) {
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
    auto n = indexer.nToOne(extent);
    auto n_VL = VL * ((n-1)/ VL + 1);
    auto dn = n_VL - n;
    auto extent_VL = extent;
    extent_VL[ND-1] += dn;
    return CppThreadContext<ND, VL>(*this, extent_VL);
  }

  template<size_t ND, size_t VL>
  bool operator==(const CppContext<ND, VL>& a, const CppContext<ND, VL>& b) {
    return (a.start == b.start) && (a.extent == b.extent);
  }

  // Now CppThreadContext
  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL>::CppThreadContext(const Context& ctx_, const Shape& pos) : ctx(ctx_) {
    auto n = ctx_.indexer.nToOne(pos);
    // Must start at a point that matches the vector length.
    assert((n % VL) == 0);
  }
  

  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL>& CppThreadContext<ND, VL>::operator++() {
    ijk += VL;
    return *this;
  }

  
  template<size_t ND, size_t VL>
  bool operator==(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return (a.ctx == b.ctx) && (a.ijk == b.ijk);
  }
  template<size_t ND, size_t VL>
  bool operator!=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return !(a == b);
  }
  
  template<size_t ND, size_t VL>
  bool operator<(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return a.ijk < b.ijk;
  }
  template<size_t ND, size_t VL>
  bool operator<=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return a.ijk <= b.ijk;
  }
  template<size_t ND, size_t VL>
  bool operator>(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return a.ijk > b.ijk;
  }
  template<size_t ND, size_t VL>
  bool operator>=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    return a.ijk >= b.ijk;
  }
  
  // template<size_t ND, size_t VL>
  // __target__ size_t CppThreadContext<ND, VL>::operator*() const {
  //   return ijk;
  // }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL> CppThreadContext<ND, VL>::begin() const {
    return CppSimdContext<ND, VL>(ctx, ijk);
  }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL> CppThreadContext<ND, VL>::end() const {
    return CppSimdContext<ND, VL>(ctx, ijk + VL);
  }

  // CppSimdContext
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL>::CppSimdContext(const Context& ctx_, const size_t& pos)
    : ctx(ctx_), idx(pos) {
  }
  
  template<size_t ND, size_t VL>
  __target__ CppSimdContext<ND, VL>& CppSimdContext<ND, VL>::operator++() {
    ++idx;
    return *this;
  }

  template<size_t ND, size_t VL>
  __target__ bool operator==(const CppSimdContext<ND,VL>& a, const CppSimdContext<ND,VL>& b) {
    return (a.ctx == b.ctx) && (a.idx == b.idx);
  }
  template<size_t ND, size_t VL>
  __target__ bool operator!=(const CppSimdContext<ND,VL>& a, const CppSimdContext<ND,VL>& b) {
    return !(a == b);
  }

  template<size_t ND, size_t VL>
  __target__ auto CppSimdContext<ND, VL>::operator*() const -> Shape {
    return ctx.indexer.oneToN(idx);
  }

  // Factory function for global iteration context
  template <size_t ND>
  __target__ CppContext<ND> MkContext(const array<size_t, ND>& shape) {
    return CppContext<ND>(shape);
  }


  // Kernel Launcher - knows the types it will be called with.
  template <size_t ND, class... FuncArgs>
  CppLauncher<ND, FuncArgs...>::CppLauncher(const ShapeT& s, const FuncT& f) : shape(s), func(f) {
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
