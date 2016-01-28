// -*- mode: C++; -*-
#ifndef SHAREDARRAY_H
#define SHAREDARRAY_H

#include "Array.h"
#include "SharedItem.h"
#include "cucall.h"

namespace {
  template<typename T, size_t ND, size_t nElem, typename... Args>
  __global__ void Construct(Array<T, ND, nElem>** self, T** elem0, Args... args) {
    *self = new Array<T, ND, nElem>(args...);
    *elem0 = (*self)->data;
  }
  template<typename T, size_t ND, size_t nElem>
  __global__ void Destruct(Array<T, ND, nElem>* self) {
    delete self;
  }
}

template<typename T, size_t ND, size_t nElem>
class SharedItem<Array<T,ND,nElem>>
/*template <typename T, size_t ND, size_t nElem>
  class SharedArray*/ : public Array<T, ND, nElem>
{
  typedef Array<T, ND, nElem> Super;
  Super* device;
  T* dev_data;
public:
  template<typename... Args>
  SharedItem(Args... args) {
    void** tmp;
    CUDA_SAFE_CALL(cudaMallocManaged(&tmp, 2*sizeof(void*), cudaMemAttachGlobal));
    Super** tmp_this = reinterpret_cast<Super**>(tmp);
    T** tmp_elem0 = reinterpret_cast<T**>(tmp+1);
    Construct<<<1,1>>>(tmp_this, tmp_elem0, args...);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    device = *tmp_this;
    dev_data = *tmp_elem0;
    CUDA_SAFE_CALL(cudaFree(tmp));
  }
  ~SharedItem() {
    Destruct<<<1,1>>>(this);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  Super* Host() {
    return this;
  }
  Super* Device() {
    return device;
  }
  
  const Super* Host() const {
    return this;
  }
  const Super* Device() const {
    return device;
  }
  void H2D() {
    CUDA_SAFE_CALL(cudaMemcpy(dev_data,
			      this->data,
			      sizeof(T),
			      cudaMemcpyHostToDevice));
  }
  void D2H() {
    CUDA_SAFE_CALL(cudaMemcpy(this->data,
			      dev_data,
			      sizeof(T),
			      cudaMemcpyDeviceToHost));
  }

  static void SwapDevicePointers(SharedItem& a, SharedItem&b) {
    Super* tmp = a.device;
    a.device = b.device;
    b.device = tmp;
  }
};


#endif
