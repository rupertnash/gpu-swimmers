// -*- mode: C++; -*-
#ifndef TARGETPP_H
#define TARGETPP_H

#include "_target.h"
#include "array.h"

// C++ style malloc
template<typename T>
void targetMalloc(T*& ptr, const size_t nbytes) {
  targetMalloc(reinterpret_cast<void**>(&ptr), nbytes);
}

template<size_t, size_t>
struct CudaThreadContext;
template<size_t, size_t>
struct CudaSimdContext;

template<size_t ND = 1, size_t VL = VVL>
struct CudaContext {
  typedef array<size_t, ND> Shape;
  
  __target__ CudaContext(const Shape& n) : extent(n) {
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

  __target__ CudaContext(const size_t n) : CudaContext(Shape(n)) {
    static_assert(ND == 1, "Context construction from integer only allowed in 1D");
  }
  
  __target__ CudaThreadContext<ND, VL> begin() const {
    return CudaThreadContext<ND, VL>(*this, start);
  }
  __target__ CudaThreadContext<ND, VL> end() const {
    return CudaThreadContext<ND, VL>(*this, finish);
  }
  
  const Shape extent;
  Shape start, finish;
};

template <size_t ND>
__target__ CudaContext<ND> MkContext(const array<size_t, ND>& shape) {
  return CudaContext<ND>(shape);
}

template<size_t ND = 1, size_t VL = VVL>
struct CudaThreadContext {
  typedef CudaContext<ND, VL> Parent;
  typedef typename Parent::Shape Shape;
  __target__ CudaThreadContext(const Parent& ctx_, const Shape& pos) : ctx(ctx_), idx(pos) {
  }
  
  __target__ CudaThreadContext& operator++() {
    idx[ND-1] = ctx.extent[ND-1];
    return *this;
  }
  
  __target__ bool operator!=(const CudaThreadContext& other) const {
    if (&ctx != &other.ctx)
      return true;  
    if (idx != other.idx)
      return true;
    return false;
  }
  
  __target__ const CudaThreadContext& operator*() const {
    return *this;
  }

  __target__ CudaSimdContext<ND, VL> begin() const {
    return CudaSimdContext<ND, VL>(*this, 0);
  }
  
  __target__ CudaSimdContext<ND, VL> end() const {
    return CudaSimdContext<ND, VL>(*this, VL);
  }
  
  const Parent& ctx;
  Shape idx;
};

template<size_t ND = 1, size_t VL = VVL>
struct CudaSimdContext {
  typedef CudaThreadContext<ND, VL> Parent;
  typedef typename Parent::Shape Shape;
  
  __target__ CudaSimdContext(const Parent& ctx_, const size_t& pos)
    : ctx(ctx_), idx(pos) {
  }
  
  __target__ CudaSimdContext& operator++() {
    ++idx;
    return *this;
  }

  __target__ bool operator!=(const CudaSimdContext& other) const {
    if (&ctx != &other.ctx)
      return true;
    if (idx != other.idx)
      return true;
    return false;
  }

  __target__ Shape operator*() const {
    Shape ans(ctx.idx);
    ans[ND-1] += idx;
    return ans;
  }

  const CudaThreadContext<ND, VL>& ctx;
  size_t idx;
};

#define FOR_TLP2(threadCtx, N) \
  for(auto threadCtx: MkContext(N))

#define FOR_TLP(N) FOR_TLP2(__targetThreadCtx, N)

#define FOR_ILP2(threadCtx, i) \
  for(auto i: threadCtx)

#define FOR_ILP(i) FOR_ILP2(__targetThreadCtx, i)

// For achieving a similar effect to :
// template<class... ParameterPack>
// struct example {
//   typedef ParameterPack Args;
// };
template<class... Args>
struct variadic_typedef {
};

// Return and argument types for functions
template<class FuncT>
struct function_traits {
};

// Main partial specialisation
template<class R, class... Args>
struct function_traits <R(Args...)> {
  typedef R(function_type)(Args...);
  typedef R return_type;
  typedef variadic_typedef<Args...> args_type;
};

// Kernel Launcher - knows the types it will be called with.
template <size_t ND, class... FuncArgs>
struct Launcher {
  typedef array<size_t, ND> ShapeT;
  
  typedef void (*FuncT)(FuncArgs...);

  const ShapeT& shape;
  const FuncT& func;
  
  dim3 nBlocks, blockShape;
  
  Launcher(const ShapeT& s, const FuncT f) : shape(s), func(f) {
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
  
  void operator()(FuncArgs... args) {
    func<<<nBlocks, blockShape>>>(args...);
  }
};

// Partial specialisation that uses the variadic pseudo typedef to unpack the args.
template <size_t ND, class... FuncArgs>
struct Launcher<ND, variadic_typedef<FuncArgs...> > : public Launcher<ND, FuncArgs...>
{
  using Launcher<ND, FuncArgs...>::Launcher;
};


// Factory function for Launchers.
//
// Uses the function traits and the specialisation on the variadic
// pseudo typedef to construct the right type of launcher
template<class FuncT, class ShapeT>
Launcher<ShapeT::size(), typename function_traits<FuncT>::args_type>
targetLaunch(FuncT* f, ShapeT shape) {
  return  Launcher<ShapeT::size(), typename function_traits<FuncT>::args_type>(shape, f);
}

#endif
