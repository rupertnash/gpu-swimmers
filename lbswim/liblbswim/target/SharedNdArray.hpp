// -*- mode: C++; -*-
#ifndef TARGET_SHAREDNDARRAY_HPP
#define TARGET_SHAREDNDARRAY_HPP

#include "SharedNdArray.h"
#include "targetpp.h"

namespace target {
  
template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Reset() {
  host = new SharedType();
  target::malloc(device);
  target::copyIn(device,
		 host);
  device_data = nullptr;
  raw_device_data = nullptr;
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Steal(const SharedItem& other) {
  host = other.host;
  device = other.device;
  device_data = other.device_data;
  raw_device_data = other.raw_device_data;
  dataSize = other.dataSize;
}

template<typename T, size_t ND, size_t nElem>
void SharedItem< NdArray<T, ND, nElem> >::Free() {
  
  target::free(raw_device_data);
  raw_device_data = nullptr;
  device_data = nullptr;
  
  target::free(device);
  device = nullptr;
  
  delete host;
  host = nullptr;
  
  dataSize = 0;
}

template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::SharedItem() {
  Reset();
}

template<typename T, size_t ND, size_t nElem>
SharedItem< NdArray<T, ND, nElem> >::SharedItem(const ShapeType& shape) {
  // Construct the master copy
  host = new SharedType(shape);
  // alloc the device main obj
  target::malloc(device);

  /*
    element_pitch = ((indexer.size - 1)/MAX_VVL + 1) * MAX_VVL;
    const auto element_pitch_bytes = element_pitch * sizeof(T);
    const auto unpadded_buffer_size_bytes = nElem * element_pitch_bytes;
    buffer_size_bytes = unpadded_buffer_size_bytes + Alignment();
    raw_data = std::malloc(buffer_size_bytes);
    void* tmp = raw_data;
    assert(std::align(Alignment(), unpadded_buffer_size_bytes, tmp, buffer_size_bytes) != NULL);
    data = static_cast<T*>(tmp);
  */

  // alloc the device data array
  target::malloc(raw_device_data, host->buffer_size_bytes);
  // align it
  void* devdata = raw_device_data;
  assert(std::align(host->Alignment(), host->element_pitch * sizeof(T) * nElem, devdata, host->buffer_size_bytes) != NULL);
  device_data = static_cast<T*>(devdata);

  // set the copy size
  dataSize = host->element_pitch * nElem;

  // prepare an array that points to the device data
  SharedType tmp;
  tmp.indexer = host->indexer;
  tmp.buffer_size_bytes = host->buffer_size_bytes;
  // This should be null as we manage the memory for the device.
  tmp.raw_data = nullptr;
  tmp.data = device_data;
  tmp.element_pitch = host->element_pitch;
  
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
NdArray<T, ND, nElem>* SharedItem< NdArray<T, ND, nElem> >::Device() {
  return device;
}
 
template<typename T, size_t ND, size_t nElem>
const NdArray<T, ND, nElem>& SharedItem< NdArray<T, ND, nElem> >::Host() const {
  return *host;
}
template<typename T, size_t ND, size_t nElem>
const NdArray<T, ND, nElem>* SharedItem< NdArray<T, ND, nElem> >::Device() const {
  return device;
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
}
#endif // SHAREDNDARRAY_HPP
