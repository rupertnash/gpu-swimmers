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

// Forward declare
template <typename T, size_t ND, size_t nElem = 1>
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

template<typename T, size_t ND, size_t nElem>
__targetBoth__ typename NdArray<T, ND, nElem>::SubType Helper(NdArray<T, ND, nElem>& self, const size_t i) {
  NdArray<T, ND-1, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  ans.owner = false;
  ans.indexer = decltype(ans.indexer)::ReduceFrom(self.indexer);
  return ans;
}

template<typename T, size_t nElem>
__targetBoth__ typename NdArray<T, 1, nElem>::SubType Helper(NdArray<T, 1, nElem>& self, const size_t i) {
  ElemWrapper<T, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  // Refactor to allow layout switching
  ans.stride = self.indexer.size;
  return ans;
}
template<typename T, size_t ND, size_t nElem>
__targetBoth__ typename NdArray<T, ND, nElem>::ConstSubType Helper(const NdArray<T, ND, nElem>& self, const size_t i) {
  NdArray<T, ND-1, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  ans.owner = false;
  ans.indexer = decltype(ans.indexer)::ReduceFrom(self.indexer);
  return ans;
}

template<typename T, size_t nElem>
__targetBoth__ typename NdArray<T, 1, nElem>::ConstSubType Helper(const NdArray<T, 1, nElem>& self, const size_t i) {
  ElemWrapper<const T, nElem> ans;
  ans.data = self.data + i * self.indexer.strides[0];
  // Refactor to allow layout switching
  ans.stride = self.indexer.size;
  return ans;
}

// Core array class.
template <typename T, size_t ND, size_t nElem>
struct NdArray
{
  typedef T ElemType;
  typedef SpaceIndexer<ND> IdxType;
  typedef typename IdxType::ShapeType ShapeType;
  typedef ElemWrapper<T, nElem> WrapType;
  typedef ElemWrapper<const T, nElem> ConstWrapType;
  //typedef std::shared_ptr<T> Ptr;
  
  typedef typename std::conditional<ND==1, WrapType, NdArray<T, ND-1, nElem> >::type SubType;
  typedef typename std::conditional<ND==1, ConstWrapType, const NdArray<T, ND-1, nElem> >::type ConstSubType;
  
  IdxType indexer;
  // Pointer to our zeroth element
  T* data;
  // Flag telling if this instance owns the data
  bool owner;
    
  // struct iterator : public std::iterator<std::bidirectional_iterator_tag, WrapType> {
  //   NdArray& container;
  //   size_t ijk;
  //   iterator(NdArray& cont, const ShapeType& startPos) : container(cont), ijk(cont.indexer(startPos))
  //   {
  //   }

  //   WrapType operator *()
  //   {
  //     WrapType ans;
  //     return Get<T,ND,nElem>(container, ijk);
  //   }
    
  //   bool operator !=(const iterator& other)const
  //   {
  //     if (container != other.container)
  // 	return true;
  //     if (ijk != other.ijk)
  // 	return true;
  //     return false;
  //   }
    
  //   iterator& operator ++()
  //   {
  //     ++ijk;
  //     return *this;
  //   }
  //   iterator& operator --()
  //   {
  //     --ijk;
  //     return *this;
  //   }
  // };
  __targetBoth__ size_t nElems() const {
    return nElem;
  }
  
  __targetBoth__ size_t nDims() const {
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
  __targetBoth__ NdArray() : indexer(ShapeType()), data(nullptr), owner(false)
  {
  }
  // Construct a given shape array
  __targetBoth__ NdArray(const ShapeType& shape) : indexer(shape), owner(true)
  {
    size_t total_size = nElem * indexer.size;
    data = new T[total_size];
  }

  NdArray(const NdArray& other) : indexer(other.indexer), data(other.data), owner(false)
  {
  }
  
  NdArray& operator=(const NdArray& other) {
    if (owner) 
      delete[] data;
    
    indexer = other.indexer;
    data = other.data;
    owner = false;
    return *this;
  }

  NdArray(NdArray&& other) : indexer(other.indexer), data(other.data), owner(other.owner) 
  {
    other.indexer = IdxType();
    other.data = nullptr;
    other.owner = false;
  }
  
  NdArray& operator=(NdArray&& other) {
    if (owner) 
      delete[] data;
    
    indexer = other.indexer;
    data = other.data;
    owner = other.owner;
    
    other.indexer = IdxType();
    other.data = nullptr;
    other.owner = false;
    
    return *this;
  }
  
  __targetBoth__ ~NdArray() {
    if (owner) {
      delete[] data;
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
    ans.stride = indexer.size;
    return ans;
  }

  __targetBoth__ ConstSubType operator[](size_t i) const {
    return Helper(*this, i);
  }
  
  __targetBoth__ ConstWrapType operator[](const ShapeType& idx) const {
    ConstWrapType ans;
    ans.data = data + indexer.nToOne(idx);
    // Refactor to allow layout switching
    ans.stride = indexer.size;
    return ans;
  }
  
  // iterator begin() {
  //   ShapeType zero = {};
  //   return iterator(*this, zero);
  // }
  // iterator end() {
  //   return iterator(*this, indexer.shape);
  // }
  __targetBoth__ bool operator !=(const NdArray& other) {
    return data != other.data;
  }
};

#endif
