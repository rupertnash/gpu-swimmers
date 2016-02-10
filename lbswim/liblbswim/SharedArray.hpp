// -*- mode: C++; -*-
#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include "SharedArray.h"

#include "target/targetpp.h"

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::Reset() {
  host = new SharedType();
  target::malloc(device);
  target::copyIn(device,
		 host);
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
  target::free(device_data);
  target::free(device);
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
  target::copyIn(device_data,
		 host->data,
		 dataSize);
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< Array<T, ND, nElem> >::D2H() {
  target::copyOut(host->data,
		  device_data,
		  dataSize);
}

#endif // SHAREDARRAY_HPP
