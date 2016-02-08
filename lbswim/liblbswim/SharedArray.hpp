// -*- mode: C++; -*-
#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include "SharedArray.h"

#include "targetpp.h"

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::Reset() {
  host = new SharedType();
  targetMalloc(device, sizeof(SharedType));
  copyToTarget(device,
	       host,
	       sizeof(SharedType));
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
  targetFree(device_data);
  targetFree(device);
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
  targetMalloc(device, sizeof(SharedType));

  dataSize = host->Size() * nElem * sizeof(T);
  targetMalloc(device_data, dataSize);
  
  SharedType tmp;
  tmp.indexer = host->indexer;
  tmp.data = device_data;
  tmp.owner = false;

  copyToTarget(device,
	       &tmp,
	       sizeof(SharedType));
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
Array<T, ND, nElem>& SharedItem< Array<T, ND, nElem> >::Host() {
  return *host;
}

template<typename T, size_t ND, size_t nElem>
Array<T, ND, nElem>& SharedItem< Array<T, ND, nElem> >::Device() {
  return *device;
}
 
template<typename T, size_t ND, size_t nElem>
const Array<T, ND, nElem>& SharedItem< Array<T, ND, nElem> >::Host() const {
  return *host;
}
template<typename T, size_t ND, size_t nElem>
const Array<T, ND, nElem>& SharedItem< Array<T, ND, nElem> >::Device() const {
  return *device;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::H2D() {
  copyToTarget(device_data,
	       host->data,
	       dataSize);
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::D2H() {
  copyFromTarget(host->data,
		 device_data,
		 dataSize);
}

#endif // SHAREDARRAY_HPP
