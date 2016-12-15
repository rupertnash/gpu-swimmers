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
    data_size_bytes = 0;
  }

  template<typename T, size_t ND, size_t nElem>
  void SharedItem< NdArray<T, ND, nElem> >::Steal(const SharedItem& other) {
    host = other.host;
    device = other.device;
    device_data = other.device_data;
    raw_device_data = other.raw_device_data;
    data_size_bytes = other.data_size_bytes;
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
  
    data_size_bytes = 0;
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

    // alloc the device data array
    target::malloc(raw_device_data, host->raw_buffer_size_bytes);

    // align it
    device_data = host->indexer.AlignRawStorage(raw_device_data);
    
    // set the copy size
    data_size_bytes = host->indexer.MinStorageSize();
    
    // prepare an array that points to the device data
    SharedType tmp;
    
    tmp.shape = host->shape;
    tmp.size = host->size;
    tmp.indexer = host->indexer;

    tmp.data = device_data;
    tmp.buffer_size = host->buffer_size;
    
    // This should be null as we manage the memory for the device.
    tmp.raw_data = nullptr;
    tmp.raw_buffer_size_bytes = host->raw_buffer_size_bytes;
    
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
    copyToTarget(device_data,
		 host->data,
		 data_size_bytes);
  }

  template<typename T, size_t ND, size_t nElem>
  void SharedItem< NdArray<T, ND, nElem> >::D2H() {
    copyFromTarget(host->data,
		   device_data,
		   data_size_bytes);
  }
}
#endif // SHAREDNDARRAY_HPP
