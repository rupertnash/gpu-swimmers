// -*- mode: C++; -*-
#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include "SharedArray.h"

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
template<typename... Args>
SharedItem< Array<T, ND, nElem> >::SharedItem(Args... args) : Array<T, ND, nElem>(args...) {
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

template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem> >::~SharedItem() {
  Destruct<<<1,1>>>(this);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename T, size_t ND, size_t nElem>
typename SharedItem< Array<T, ND, nElem> >::Super* SharedItem< Array<T, ND, nElem> >::Host() {
  return this;
}

template<typename T, size_t ND, size_t nElem>
typename SharedItem< Array<T, ND, nElem> >::Super* SharedItem< Array<T, ND, nElem> >::Device() {
  return device;
}
 
template<typename T, size_t ND, size_t nElem>
const typename SharedItem< Array<T, ND, nElem> >::Super* SharedItem< Array<T, ND, nElem> >::Host() const {
  return this;
}
template<typename T, size_t ND, size_t nElem>
const typename SharedItem< Array<T, ND, nElem> >::Super* SharedItem< Array<T, ND, nElem> >::Device() const {
  return device;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::H2D() {
  CUDA_SAFE_CALL(cudaMemcpy(dev_data,
			    this->data,
			    sizeof(T) * this->nElems() * this->Size(),
			    cudaMemcpyHostToDevice));
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::D2H() {
  CUDA_SAFE_CALL(cudaMemcpy(this->data,
			    dev_data,
			    sizeof(T) * this->nElems() * this->Size(),
			    cudaMemcpyDeviceToHost));
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::SwapDevicePointers(SharedItem& a, SharedItem&b) {
  Super* tmp = a.device;
  a.device = b.device;
  b.device = tmp;
}

#endif // SHAREDARRAY_HPP
