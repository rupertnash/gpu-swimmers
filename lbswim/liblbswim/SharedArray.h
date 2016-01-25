// -*- mode: C++; -*-
#ifndef SHAREDARRAY_H
#define SHAREDARRAY_H

#include "cucall.h"

template<typename T>
struct SharedArray
{
  T* host;
  T* device;
  size_t size;
  SharedArray(size_t n);
  ~SharedArray();
  void H2D();
  void D2H();
};

template <typename T>
SharedArray<T>::SharedArray(size_t n) : size(n) {
  host = new T[n];
  CUDA_SAFE_CALL(cudaMalloc(&device, n * sizeof(T)));
}
template <typename T>
SharedArray<T>::~SharedArray() {
  delete[] host;
  CUDA_SAFE_CALL(cudaFree(device));
}
template <typename T>
void SharedArray<T>::H2D() {
  CUDA_SAFE_CALL(cudaMemcpy(device,
			    host,
			    size * sizeof(T),
			    cudaMemcpyHostToDevice));
}
template <typename T>
void SharedArray<T>::D2H() {
  CUDA_SAFE_CALL(cudaMemcpy(host,
			    device,
			    size * sizeof(T),
			    cudaMemcpyDeviceToHost));
}

#endif
