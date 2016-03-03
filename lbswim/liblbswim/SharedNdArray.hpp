// -*- mode: C++; -*-
#ifndef SHAREDNDARRAY_HPP
#define SHAREDNDARRAY_HPP

#include "SharedNdArray.h"

#include "target/targetpp.h"

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Reset() {
  host = new SharedType();
  target::malloc(device);
  target::copyIn(device,
		 host);
  device_data = nullptr;
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Steal(const SharedItem& other) {
  host = other.host;
  device = other.device;
  device_data = other.device_data;
  dataSize = other.dataSize;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Free() {
  target::free(device_data);
  target::free(device);
  delete host;
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::SharedItem() {
  Reset();
}

template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::SharedItem(const ShapeType& shape) {
  host = new SharedType(shape);
  target::malloc(device);

  dataSize = host->Size() * nElem;
  target::malloc(device_data, dataSize);
  
  SharedType tmp;
  tmp.indexer = host->indexer;
  tmp.data = device_data;
  tmp.owner = false;

  target::copyIn(device,
		 &tmp);
}

// Move constructor
template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::SharedItem(SharedItem&& other) {
  // Steal other's resources
  Steal(other);
  // Reset other's members
  other.Reset();
}

// Move assign
template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem > >& SharedItem< NdArray<T, ND, nElem> >::operator=(SharedItem&& other) {
  // Free my resources
  Free();
  // Steal other's resources
  Steal(other);  
  // Reset other's members
  other.Reset();
  return *this;
}

template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::~SharedItem() {
  Free();
}

template<typename T, size_t ND, size_t nElem>
NdArray<T, ND, nElem>& SharedItem< NdArray<T, ND, nElem> >::Host() {
  return *host;
}

template<typename T, size_t ND, size_t nElem>
NdArray<T, ND, nElem>& SharedItem< NdArray<T, ND, nElem> >::Device() {
  return *device;
}
 
template<typename T, size_t ND, size_t nElem>
const NdArray<T, ND, nElem>& SharedItem< NdArray<T, ND, nElem> >::Host() const {
  return *host;
}
template<typename T, size_t ND, size_t nElem>
const NdArray<T, ND, nElem>& SharedItem< NdArray<T, ND, nElem> >::Device() const {
  return *device;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::H2D() {
  target::copyIn(device_data,
		 host->data,
		 dataSize);
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::D2H() {
  target::copyOut(host->data,
		  device_data,
		  dataSize);
}

#endif // SHAREDNDARRAY_HPP
