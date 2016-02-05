// -*- mode: C++; -*-
#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include "SharedArray.h"

#include "cucall.h"

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::Reset() {
  host = new SharedType();
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(SharedType)));
  CUDA_SAFE_CALL(cudaMemcpy(device,
			    host,
			    sizeof(SharedType),
			    cudaMemcpyHostToDevice));
  device_data = nullptr;
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::Steal(const SharedItem& other) {
  host = other.host;
  device = other.device;
  device_data = other.device_data;
  dataSize = other.dataSize;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::Free() {
  CUDA_SAFE_CALL(cudaFree(device_data));
  CUDA_SAFE_CALL(cudaFree(device));
  delete host;
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem> >::SharedItem() {
  Reset();
}

template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem> >::SharedItem(const ShapeType& shape) {
  host = new SharedType(shape);
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(SharedType)));

  dataSize = host->Size() * nElem * sizeof(T);
  CUDA_SAFE_CALL(cudaMalloc(&device_data, dataSize));
  
  SharedType tmp;
  tmp.indexer = host->indexer;
  tmp.data = device_data;
  tmp.owner = false;

  CUDA_SAFE_CALL(cudaMemcpy(device,
			    &tmp,
			    sizeof(SharedType),
			    cudaMemcpyHostToDevice));
}

// Move constructor
template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem> >::SharedItem(SharedItem&& other) {
  // Steal other's resources
  Steal(other);
  // Reset other's members
  other.Reset();
}

// Move assign
template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem > >& SharedItem< Array<T, ND, nElem> >::operator=(SharedItem&& other) {
  // Free my resources
  Free();
  // Steal other's resources
  Steal(other);  
  // Reset other's members
  other.Reset();
  return *this;
}

template<typename T, size_t ND, size_t nElem>
SharedItem< Array<T, ND, nElem> >::~SharedItem() {
  Free();
}

template<typename T, size_t ND, size_t nElem>
Array<T, ND, nElem>* SharedItem< Array<T, ND, nElem> >::Host() {
  return host;
}

template<typename T, size_t ND, size_t nElem>
Array<T, ND, nElem>* SharedItem< Array<T, ND, nElem> >::Device() {
  return device;
}
 
template<typename T, size_t ND, size_t nElem>
const Array<T, ND, nElem>* SharedItem< Array<T, ND, nElem> >::Host() const {
  return host;
}
template<typename T, size_t ND, size_t nElem>
const Array<T, ND, nElem>* SharedItem< Array<T, ND, nElem> >::Device() const {
  return device;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::H2D() {
  CUDA_SAFE_CALL(cudaMemcpy(device_data,
			    host->data,
			    dataSize,
			    cudaMemcpyHostToDevice));
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::D2H() {
  CUDA_SAFE_CALL(cudaMemcpy(host->data,
			    device_data,
			    dataSize,
			    cudaMemcpyDeviceToHost));
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::SwapDevicePointers(SharedItem& a, SharedItem&b) {
  SharedType* tmp = a.device;
  a.device = b.device;
  b.device = tmp;
  // Swap device_data?
}

#endif // SHAREDARRAY_HPP
