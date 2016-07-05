// -*- mode: C++; -*-

// Multidimensional array template class, where each element is a
// vector. Allows configuration of the memory layout at
// compile time.

#ifndef NDARRAY_H
#define NDARRAY_H

#include "func_attr.h"
#include "array.h"

#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <assert.h>

#include "Nd.h"

#if defined(NOSTDALIGN) || defined(TARGET_MODE_CUDA)
// Older versions of GCC don't have std::align in their standard
// library, even in C++11 mode. Provide an unoptimal version to cover
// this case.
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

namespace target {
  
  // Forward declarations
  template <typename T, size_t ND,
	    size_t nElem = 1,
	    size_t ALIGN = 0,
	    size_t MAX_VVL = 16>
  class NdArray;
  
  template<class, size_t>
  struct VectorView;
  
  // Wrapper for a single element vector of an NdArray.
  template <typename T, size_t nElem>
  struct ElemWrapper
  {
    // The pointer to our zeroth element
    T* data;
    // number of elements to the next element.
    size_t stride;

  public:
    //ElemWrapper(T* d, size_t str) : data(d), stride(str) {}
    // Get an element
    __targetBoth__ T& operator[](const size_t& i) {
      return data[i * stride];
    }
    // Get a const element
    __targetBoth__ const T& operator[](const size_t& i) const {
      return data[i * stride];
    }
  };

  class SubscriptHelper;
  
  template <class T>
  class SharedItem;
  
  // Core array class.
  // ALIGN = byte boundary required by the vector instructions you want to use. 0 => alignof(T)
  // MAX_VVL = the max VVL you want to use
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  class NdArray
  {
  public:
    __targetBoth__ constexpr static size_t Alignment();
    
    typedef T ElemType;
    typedef SpaceIndexer<ND> IdxType;
    typedef typename IdxType::ShapeType ShapeType;
    typedef ElemWrapper<T, nElem> WrapType;
    typedef ElemWrapper<const T, nElem> ConstWrapType;
  
    typedef typename std::conditional<ND==1, WrapType, NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> >::type SubType;
    typedef typename std::conditional<ND==1, ConstWrapType, const NdArray<T, ND-1, nElem, ALIGN, MAX_VVL> >::type ConstSubType;
    
    __targetBoth__ bool OwnsData() const;
  
    __targetBoth__ constexpr static size_t nElems();
  
    __targetBoth__ constexpr static size_t MaxVVL();
  
    __targetBoth__ constexpr static size_t nDims();
    __targetBoth__ const ShapeType& Shape() const;
    __targetBoth__ const ShapeType& Strides() const;
    __targetBoth__ size_t Size() const;
    __targetBoth__ size_t Pitch() const;
    
    __targetBoth__ T* Data();
    __targetBoth__ const T* Data() const;

    // Default constructor
    __targetBoth__ NdArray();
    // Construct a given shape array
    __targetBoth__ NdArray(const ShapeType& shape);

    __targetBoth__ NdArray(const NdArray& other);
  
    __targetBoth__ NdArray& operator=(const NdArray& other);

    __targetBoth__ NdArray(NdArray&& other);
  
    __targetBoth__ NdArray& operator=(NdArray&& other);
  
    __targetBoth__ ~NdArray();
  
    // Subscript to return an array with dimensionality ND-1 or an
    // ElementWrapper, as appropriate.
    __targetBoth__ SubType operator[](size_t i);
    __targetBoth__ WrapType operator[](const ShapeType& idx);

    __targetBoth__ ConstSubType operator[](size_t i) const;
    __targetBoth__ ConstWrapType operator[](const ShapeType& idx) const;
  
    __targetBoth__ ElemType& operator()(const size_t i, const size_t d);
    __targetBoth__ const ElemType& operator()(const size_t i, const size_t d) const;

    __targetBoth__ ElemType& operator()(const size_t i, const size_t j, const size_t d);
    __targetBoth__ const ElemType& operator()(const size_t i, const size_t j, const size_t d) const;
  
    __targetBoth__ bool operator !=(const NdArray& other);

    template<size_t ViewVL>
    __targetBoth__ VectorView<NdArray, ViewVL> GetVectorView(size_t ijk);
    template<size_t ViewVL>
    __targetBoth__ VectorView<const NdArray, ViewVL> GetVectorView(size_t ijk) const;
    
  private:
    IdxType indexer;
    // Pointer to our zeroth element
    T* data;
    // Underlying, un-aligned data storage & size
    // non-null => this instance owns the data & vice-versa
    size_t buffer_size_bytes;
    void* raw_data;
    // Stride (in units of sizeof(T) bytes) between the elements of a contained item
    size_t element_pitch;
    
    friend class SubscriptHelper;
    friend class SharedItem<NdArray>;
  };

  template<class AT, size_t VL>
  struct VectorView {
    typedef AT ArrayT;
    typedef typename AT::ElemType ElemType;
    typedef typename AT::WrapType WrapType;

    WrapType zero;
    __targetBoth__ WrapType operator[](size_t i);
    __targetBoth__ ElemType& operator()(size_t i, size_t d);
    __targetBoth__ const ElemType& operator()(size_t i, size_t d) const;
  };

}

#include "NdArrayImpl.hpp"
#endif
