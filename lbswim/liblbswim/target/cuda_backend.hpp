#ifndef TARGETPP_CUDA_BACKEND_HPP
#define TARGETPP_CUDA_BACKEND_HPP

#include "./cuda_backend.h"

namespace target {

  template<size_t ND, size_t VL>
  __target__ CudaContext<ND,VL>::CudaContext(const Shape& n) : extent(n),
							       indexer(n) {
    start = VL * (blockIdx.x * blockDim.x + threadIdx.x);
    // smaller of start+VL and indexer.size
    finish = ((start + VL) < indexer.size) ?
      (start + VL) :
      indexer.size;
  }

  template<size_t ND, size_t VL>
  __target__ CudaContext<ND,VL>::CudaContext(const size_t n) : CudaContext(Shape(n)) {
    static_assert(ND == 1, "Context construction from integer only allowed in 1D");
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaThreadContext<ND, VL> CudaContext<ND,VL>::begin() const {
    return CudaThreadContext<ND, VL>(*this, start);
  }
  template<size_t ND, size_t VL>
  __target__ CudaThreadContext<ND, VL> CudaContext<ND,VL>::end() const {
    return CudaThreadContext<ND, VL>(*this, finish);
  }
  

  // Now CudaThreadContext
  template<size_t ND, size_t VL>
  __target__ CudaThreadContext<ND, VL>::CudaThreadContext(const Parent& ctx_, const size_t& pos) : ctx(ctx_), ijk(pos) {
    // Must start at a point that matches the vector length.
    assert((ijk % VL) == 0);
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaThreadContext<ND, VL>& CudaThreadContext<ND, VL>::operator++() {
    ijk += VL;
    return *this;
  }
  
  template<size_t ND, size_t VL>
  __target__ bool CudaThreadContext<ND, VL>::operator!=(const CudaThreadContext& other) const {
    if (&ctx != &other.ctx)
      return true;  
    if (ijk != other.ijk)
      return true;
    return false;
  }
  
  template<size_t ND, size_t VL>
  __target__ bool operator<(const CudaThreadContext<ND, VL>& a, const CudaThreadContext<ND, VL>& b) {
    return a.ijk < b.ijk;
  }

  template<size_t ND, size_t VL>
  __target__ auto CudaThreadContext<ND, VL>::operator[](size_t ilp_idx) const -> Shape {
    return ctx.indexer.oneToN(ijk + ilp_idx);
  }

  template<size_t ND, size_t VL>
  __target__ const CudaThreadContext<ND, VL>& CudaThreadContext<ND, VL>::operator*() const {
    return *this;
  }
  template<size_t ND, size_t VL>
  template <class ArrayT>
  __target__ VectorView<ArrayT, VL> CudaThreadContext<ND, VL>::GetCurrentElements(ArrayT* arr) {
    //static_assert(arr.MaxVVL() >= VL, "Array not guaranteed to work with this vector length");
    assert(ijk % VL == 0);
    
    VectorView<ArrayT, VL> ans;
    ans.zero.data = arr->data + ijk;
    ans.zero.stride = arr->element_pitch;
    return ans;
  }
  
  template<size_t ND, size_t VL>
  __target__ auto CudaThreadContext<ND, VL>::GetNdIndex(size_t i) const -> Shape {
    return ctx.indexer.oneToN(ijk + i);
  }
  template<class AT, size_t VL>
  __target__ auto VectorView<AT, VL>::operator[](size_t i) -> WrapType {
    WrapType ans = zero;
    ans.data += i;
    return ans;
  }
  
  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  __targetBoth__ Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::ArgTypes(const Index& shape) : extent(shape) {
    static_assert(ND > 0, "Must have at least one dimension");
    auto nElem = Product(shape);
    // Round up to the next VL
    auto nElem_VL = ((nElem-1) / VL + 1) * VL;
    
    blockShape = DEFAULT_TPB;
    nBlocks = ((nElem_VL / VL) + blockShape - 1) / blockShape;
  }
  
  template<class Impl, size_t ND, size_t VL, class... Args>
  __targetEntry__ void TargetKernelEntry(const array<size_t, ND> shape, Args... args) {
    Impl kernel(shape);
    kernel.indexSpace = new CudaContext<ND,VL>(shape);
    kernel.Run(args...);
    delete kernel.indexSpace;
  }
  
  template<class Impl>
  template<size_t ND>
  template<size_t VL>
  template<class... Args>
  void Kernel<Impl>::Dims<ND>::VecLen<VL>::ArgTypes<Args...>::operator()(Args... args) {
    TargetKernelEntry<Impl, ND, VL, Args...> <<<nBlocks, blockShape>>> (extent, args...);
  }

}
#endif
