#ifndef TARGET_CPP_BACKEND_HPP
#define TARGET_CPP_BACKEND_HPP

#include "./cpp_backend.h"
#include <cassert>

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
    // Last valid ND index
    auto last = extent;
    for (size_t d = 0; d< ND; d++)
      last[d] -= 1;
    // 1D index
    auto last_ijk = indexer.nToOne(last);
    // Number of elems
    auto n = last_ijk + 1;
    // Number rounded up to the next VL
    auto n_VL = VL * ((n-1)/ VL + 1);
    // Extra
    auto dn = n_VL - n;
    // point to the first element after the VL chunked array
    last[ND-1] += 1 + dn;
    return CppThreadContext<ND, VL>(*this, last);
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
    ijk = n;
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
  
  template<size_t ND, size_t VL>
  __target__ std::ptrdiff_t operator-(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b) {
    assert(a.ctx == b.ctx);
    return (a.ijk - b.ijk) / VL;
  }
  
  template<size_t ND, size_t VL>
  __target__ CppThreadContext<ND, VL>& CppThreadContext<ND, VL>::operator+=(std::ptrdiff_t inc) {
    ijk += VL * inc;
    return *this;
  }
  
  template<size_t ND, size_t VL>
  template <class ArrayT>
  __target__ VectorView<ArrayT, VL> CppThreadContext<ND, VL>::GetCurrentElements(ArrayT* arr) {
    static_assert(ArrayT::MaxVVL() >= VL, "Array not guaranteed to work with this vector length");
    assert(ijk % VL == 0);
    
    VectorView<ArrayT, VL> ans;
    ans.zero.data = arr->data + ijk;
    //(typename ArrayT::ElemType *)__builtin_assume_aligned(arr->data + ijk, sizeof(typename ArrayT::ElemType) * VL);
    ans.zero.stride = arr->element_pitch;
    return ans;
  }
  
  template<size_t ND, size_t VL>
  auto CppThreadContext<ND, VL>::GetNdIndex(size_t i) const -> Shape {
    return ctx.indexer.oneToN(ijk + i);
  }
  template<class AT, size_t VL>
  auto VectorView<AT, VL>::operator[](size_t i) -> WrapType {
    WrapType ans = zero;
    ans.data += i;
    return ans;
  }
  
  // Factory function for global iteration context
  template <size_t ND, size_t VL>
  __target__ CppContext<ND, VL> MkContext(const array<size_t, ND>& shape) {
    return CppContext<ND, VL>(shape);
  }
  

  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  __targetBoth__ Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::ArgTypes(const Index& shape) : extent(shape) {
    static_assert(ND > 0, "Must have at least one dimension");
  }
  
  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  void Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::operator()(Args... args) {
    indexSpace = new CppContext<ND,VL>(extent);
    static_cast<Impl*>(this)->Run(args...);
    delete indexSpace;
  }
}
#endif
