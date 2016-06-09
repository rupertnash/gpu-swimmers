// -*- mode: C++; -*-

// Multidimensional array template class, where each element is a
// vector. Allows configuration of the memory layout at
// compile time.

#ifndef NDARRAY_H
#define NDARRAY_H

#include "target/func_attr.h"

#include "array.h"
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <assert.h>

#include "Nd.h"

#if defined(NOSTDALIGN) || defined(TARGET_MODE_CUDA)
namespace std {
  __targetBoth__ inline void*
  align(size_t align, size_t size, void*& ptr, size_t& space) noexcept
  {
    const auto intptr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = (intptr - 1u + align) & -align;
    const auto diff = aligned - intptr;
    if ((size + diff) > space)
      return nullptr;
    else
      {
	space -= diff;
	return ptr = reinterpret_cast<void*>(aligned);
      }
  }
}
#endif

__targetBoth__ inline void* Malloc(size_t size) {
#ifdef TARGET_DEVICE_CODE
  char* ans;
  target::malloc(ans, size);
  return static_cast<void*>(ans);
#else
  return std::malloc(size);
#endif
}
__targetBoth__ inline void Free(void* ptr) {
#ifdef TARGET_DEVICE_CODE
  char* tmp = static_cast<char*>(ptr);
  target::free<char>(static_cast<char*>(tmp));
#else
  std::free(ptr);
#endif
}

// Forward declare
template <typename T, size_t ND, size_t nElem = 1, size_t ALIGN = 0, size_t MAX_VVL = 16>
struct NdArray;

// Wrapper for a single element vector of an NdArray.
template <typename T, size_t nElem>
struct ElemWrapper
{
  // The pointer to our zeroth element
  T* data;
  // number of elements to the next element.
  size_t stride;

public:
  // Get an element
  __targetBoth__ T& operator[](const size_t& i) {
    return data[i * stride];
  }
  // Get a const element
 __targetBoth__ const T& operator[](const size_t& i) const {
    return data[i * stride];
  }
};

template<typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
__targetBoth__ typename NdArray<T, ND, nElem, ALIGN, MAX_VVL>::SubType Helper(NdArray<T, ND, nElem, ALIGN, MAX_VVL>& self, const size_t i) {
  NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> ans;
  const auto offset = i * self.indexer.strides[0];
  ans.data = self.data + offset;
  ans.buffer_size_bytes = self.buffer_size_bytes - offset*sizeof(T);
  ans.raw_data = nullptr;
  ans.element_pitch = self.element_pitch;
  ans.indexer = decltype(ans.indexer)::ReduceFrom(self.indexer);
  return ans;
}

template<typename T, size_t nElem, size_t ALIGN, size_t MAX_VVL>
__targetBoth__ typename NdArray<T, 1, nElem, ALIGN, MAX_VVL>::SubType Helper(NdArray<T, 1, nElem, ALIGN, MAX_VVL>& self, const size_t i) {
  ElemWrapper<T, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  // Refactor to allow layout switching
  ans.stride = self.element_pitch;
  return ans;
}
template<typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
__targetBoth__ typename NdArray<T, ND, nElem, ALIGN, MAX_VVL>::ConstSubType Helper(const NdArray<T, ND, nElem, ALIGN, MAX_VVL>& self, const size_t i) {
  NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> ans;
  const auto offset = i * self.indexer.strides[0];
  ans.data = self.data + offset;
  ans.buffer_size_bytes = self.buffer_size_bytes - offset*sizeof(T);
  ans.raw_data = nullptr;
  ans.element_pitch = self.element_pitch;
  ans.indexer = decltype(ans.indexer)::ReduceFrom(self.indexer);
  return ans;
}

template<typename T, size_t nElem, size_t ALIGN, size_t MAX_VVL>
__targetBoth__ typename NdArray<T, 1, nElem, ALIGN, MAX_VVL>::ConstSubType Helper(const NdArray<T, 1, nElem, ALIGN, MAX_VVL>& self, const size_t i) {
  ElemWrapper<const T, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  // Refactor to allow layout switching
  ans.stride = self.element_pitch;
  return ans;
}

// Core array class.
// ALIGN = byte boundary required by the vector instructions you want to use. 0 => alignof(T)
// MAX_VVL = the max VVL you want to use
template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
struct NdArray
{
  __targetBoth__ constexpr static size_t Alignment() {
    return ALIGN ? ALIGN : alignof(T);
  }
    
  //static_assert(MAX_VVL >= Alignment(), "Padding must be >= to alignment");
  typedef T ElemType;
  typedef SpaceIndexer<ND> IdxType;
  typedef typename IdxType::ShapeType ShapeType;
  typedef ElemWrapper<T, nElem> WrapType;
  typedef ElemWrapper<const T, nElem> ConstWrapType;
  //typedef std::shared_ptr<T> Ptr;
  
  typedef typename std::conditional<ND==1, WrapType, NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> >::type SubType;
  typedef typename std::conditional<ND==1, ConstWrapType, const NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> >::type ConstSubType;
  
  IdxType indexer;
  // Pointer to our zeroth element
  T* data;
  // Underlying, un-aligned data storage & size
  // non-null => this instance owns the data & vice-versa
  size_t buffer_size_bytes;
  void* raw_data;
  // Stride (in units of sizeof(T) bytes) between the elements of a contained item
  size_t element_pitch;

  __targetBoth__ bool OwnsData() const {
    return raw_data != nullptr;
  }
  
  __targetBoth__ constexpr static size_t nElems() {
    return nElem;
  }
  
  __targetBoth__ constexpr static size_t MaxVVL() {
    return MAX_VVL;
  }
  
  __targetBoth__ constexpr static size_t nDims() {
    return ND;
  }
  __targetBoth__ const ShapeType& Shape() const {
    return indexer.shape;
  }
  __targetBoth__ const ShapeType& Strides() const {
    return indexer.strides;
  }
  __targetBoth__ size_t Size() const {
    return indexer.size;
  }
  
  // Default constructor
  __targetBoth__ NdArray() : indexer(ShapeType()), data(nullptr), buffer_size_bytes(0), raw_data(nullptr), element_pitch(0)
  {
  }
  // Construct a given shape array
  __targetBoth__ NdArray(const ShapeType& shape) : indexer(shape)
  {
    element_pitch = ((indexer.size - 1)/MAX_VVL + 1) * MAX_VVL;
    const auto element_pitch_bytes = element_pitch * sizeof(T);
    const auto unpadded_buffer_size_bytes = nElem * element_pitch_bytes;
    buffer_size_bytes = unpadded_buffer_size_bytes + Alignment();
    raw_data = std::malloc(buffer_size_bytes);
    void* tmp = raw_data;
    assert(std::align(Alignment(), unpadded_buffer_size_bytes, tmp, buffer_size_bytes) != NULL);
    data = static_cast<T*>(tmp);
  }

  __targetBoth__ NdArray(const NdArray& other) : indexer(other.indexer),
						 data(other.data),
						 buffer_size_bytes(other.buffer_size_bytes),
						 raw_data(nullptr),
						 element_pitch(other.element_pitch)
  {
  }
  
  __targetBoth__ NdArray& operator=(const NdArray& other) {
    if (OwnsData()) {
      std::free(raw_data);
      raw_data = nullptr;
    }
    
    indexer = other.indexer;
    data = other.data;
    buffer_size_bytes = other.buffer_size_bytes;
    element_pitch = other.element_pitch;
    
    return *this;
  }

  __targetBoth__ NdArray(NdArray&& other) : indexer(other.indexer),
					    data(other.data),
					    buffer_size_bytes(other.buffer_size_bytes),
					    raw_data(other.raw_data),
					    element_pitch(other.element_pitch)
  {
    other.indexer = IdxType();
    other.data = nullptr;
    other.raw_data = nullptr;
  }
  
  __targetBoth__ NdArray& operator=(NdArray&& other) {
    if (OwnsData()) {
      std::free(raw_data);
      raw_data = nullptr;
    }
    
    indexer = other.indexer;
    data = other.data;
    buffer_size_bytes = other.buffer_size_bytes;
    raw_data = other.raw_data;
    element_pitch = other.element_pitch;
    
    other.indexer = IdxType();
    other.data = nullptr;
    other.raw_data = nullptr;
    
    return *this;
  }
  
  __targetBoth__ ~NdArray() {
    if (OwnsData()) {
      std::free(raw_data);
      raw_data = nullptr;
      data = nullptr;
    }
  }
  
  // Subscript to return an array with dimensionality ND-1 or an
  // ElementWrapper, as appropriate.
  __targetBoth__ SubType operator[](size_t i) {
    return Helper(*this, i);
  }
  
  __targetBoth__ WrapType operator[](const ShapeType& idx) {
    WrapType ans;
    ans.data = data + indexer.nToOne(idx);
    // Refactor to allow layout switching
    ans.stride = element_pitch;
    return ans;
  }

  __targetBoth__ ConstSubType operator[](size_t i) const {
    return Helper(*this, i);
  }
  
  __targetBoth__ ConstWrapType operator[](const ShapeType& idx) const {
    ConstWrapType ans;
    ans.data = data + indexer.nToOne(idx);
    // Refactor to allow layout switching
    ans.stride = element_pitch;
    return ans;
  }
  
  __targetBoth__ ElemType& operator()(const size_t i, const size_t d) {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[i*indexer.strides[0] + element_pitch*d];
  }
  __targetBoth__ const ElemType& operator()(const size_t i, const size_t d) const {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[i*indexer.strides[0] + element_pitch*d];
  }

  __targetBoth__ ElemType& operator()(const size_t i, const size_t j, const size_t d) {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + element_pitch*d];
  }
  __targetBoth__ const ElemType& operator()(const size_t i, const size_t j, const size_t d) const {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + element_pitch*d];
  }
  
  __targetBoth__ bool operator !=(const NdArray& other) {
    return data != other.data;
  }
};

#endif
