#ifndef TARGETPP_CUDA_BACKEND_HPP
#define TARGETPP_CUDA_BACKEND_HPP

#include "./cuda_backend.h"

namespace target {
  template<size_t ND, size_t VL>
  __target__ CudaContext<ND,VL>::CudaContext(const Shape& n) : extent(n) {
    static_assert(ND > 0, "Must have at least one dimension");
    static_assert(ND <= 3, "CUDA doesn't deal with more than 3D");

    start[0] = blockIdx.x * blockDim.x + threadIdx.x;
    if (ND > 1) {
      start[1] = blockIdx.y * blockDim.y + threadIdx.y;
      if (ND > 2) {
	start[2] = (blockIdx.z * blockDim.z + threadIdx.z);
      }
    }
    start[ND-1] *= VL;
    
    for (size_t i=0; i<ND; ++i) {
      start[i] = (start[i] > extent[i]) ? extent[i] : start[i];
      finish[i] = start[i];
    }
    finish[ND-1] = extent[ND-1];
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
  __target__ CudaThreadContext<ND, VL>::CudaThreadContext(const Parent& ctx_, const Shape& pos) : ctx(ctx_), idx(pos) {
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaThreadContext<ND, VL>& CudaThreadContext<ND, VL>::operator++() {
    idx[ND-1] = ctx.extent[ND-1];
    return *this;
  }
  
  template<size_t ND, size_t VL>  
  __target__ bool CudaThreadContext<ND, VL>::operator!=(const CudaThreadContext& other) const {
    if (&ctx != &other.ctx)
      return true;  
    if (idx != other.idx)
      return true;
    return false;
  }
  
  template<size_t ND, size_t VL>
  __target__ const CudaThreadContext<ND, VL>& CudaThreadContext<ND, VL>::operator*() const {
    return *this;
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaSimdContext<ND, VL> CudaThreadContext<ND, VL>::begin() const {
    return CudaSimdContext<ND, VL>(*this, 0);
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaSimdContext<ND, VL> CudaThreadContext<ND, VL>::end() const {
    return CudaSimdContext<ND, VL>(*this, VL);
  }
  
  // CudaSimdContext
  template<size_t ND, size_t VL>
  __target__ CudaSimdContext<ND, VL>::CudaSimdContext(const Parent& ctx_, const size_t& pos)
    : ctx(ctx_), idx(pos) {
  }
  
  template<size_t ND, size_t VL>
  __target__ CudaSimdContext<ND, VL>& CudaSimdContext<ND, VL>::operator++() {
    ++idx;
    return *this;
  }

  template<size_t ND, size_t VL>
  __target__ bool CudaSimdContext<ND, VL>::operator!=(const CudaSimdContext& other) const {
    if (&ctx != &other.ctx)
      return true;
    if (idx != other.idx)
      return true;
    return false;
  }

  template<size_t ND, size_t VL>
  __target__ auto CudaSimdContext<ND, VL>::operator*() const -> Shape {
    Shape ans(ctx.idx);
    ans[ND-1] += idx;
    return ans;
  }

  // Kernel Launcher - knows the types it will be called with.
  template <size_t ND, class... FuncArgs>
  CudaLauncher<ND, FuncArgs...>::CudaLauncher(const ShapeT& s, const FuncT f) : shape(s), func(f) {
    static_assert(ND > 0, "Must have at least one dimension");
    static_assert(ND <= 3, "CUDA doesn't deal with more than 3D");
    
    const size_t bs_x = ND == 1 ? 128 : (ND == 2 ?  8 : 4);
    const size_t bs_y = ND == 1 ?   1 : (ND == 2 ? 16 : 4);
    const size_t bs_z = ND == 1 ?   1 : (ND == 2 ?  1 : 8);

    blockShape = {bs_x, bs_y, bs_z};

    switch (ND) {
    case 1:
      nBlocks.x = ((shape[0] / VVL) + blockShape.x - 1) / blockShape.x;
      break;
    case 2:
      nBlocks.x = (shape[0] + blockShape.x - 1)/blockShape.x;
      nBlocks.y = ((shape[1] / VVL) + blockShape.y - 1) / blockShape.y;
      break;
    case 3:
      nBlocks.x = (shape[0] + blockShape.x - 1)/blockShape.x;
      nBlocks.y = (shape[1] + blockShape.y - 1)/blockShape.y;
      nBlocks.z = ((shape[2]/VVL) + blockShape.z - 1)/blockShape.z;
      break;
    default:
      break;
    }
    
    const size_t total_threads =  bs_x * bs_y * bs_z;
    static_assert(total_threads == DEFAULT_TPB, "block shape doesn't match DEFAULT_TPB");
  }
  
  template <size_t ND, class... FuncArgs>
  void CudaLauncher<ND, FuncArgs...>::operator()(FuncArgs... args) {
    func<<<nBlocks, blockShape>>>(args...);
  }

  // Partial specialisation on the pseudo typedef of the args
  // parameter pack. This subclasses the class we actually want but
  // then inherits its constructor.
  template <size_t ND, class... FuncArgs>
  struct CudaLauncher<ND, variadic_typedef<FuncArgs...> > : public CudaLauncher<ND, FuncArgs...>
  {
    // Inherit c'tor
    using CudaLauncher<ND, FuncArgs...>::CudaLauncher;
  };

  // Factory function for global iteration context
  template <size_t ND>
  __target__ CudaContext<ND> MkContext(const array<size_t, ND>& shape) {
    return CudaContext<ND>(shape);
  }
  
  // Factory function for Launchers.
  //
  // Uses the function traits and the specialisation on the variadic
  // pseudo typedef to construct the right type of launcher
  template<class FuncT, class ShapeT>
  CudaLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>
  launch(FuncT* f, ShapeT shape) {
    return CudaLauncher<ShapeT::size(), typename function_traits<FuncT>::args_type>(shape, f);
  }

}
#endif
