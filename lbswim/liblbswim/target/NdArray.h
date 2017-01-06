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
	    size_t nElem,
	    typename LayoutMetaFunc>
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

  //class SubscriptHelper;
  
  template <class T>
  class SharedItem;

  // Struct of arrays policy 
  // ALIGN = byte boundary required by the vector instructions you want to use. 0 => alignof(T)
  // MAX_VVL = the max VVL you want to use
  template<size_t ALIGN, size_t MAX_VVL>
  struct SoALayoutPolicy {
    template<typename T, size_t ND, size_t nElem>
    struct value {
    public:
      typedef T ElemType;
      typedef array<size_t, ND> ShapeType;
      
      __targetBoth__ value() : strides() , element_pitch(0) {
      }
      __targetBoth__ value(const ShapeType& shp) {
	strides[ND-1] = 1;
	for (int i = int(ND) - 2; i >= 0; --i)
	  strides[i] = strides[i+1] * shp[i+1];
	  
	size_t size = strides[0] * shp[0];
	element_pitch = ((size - 1)/MaxVVL() + 1) * MaxVVL();
      }
	
      __targetBoth__ size_t IndexToOffset(const ShapeType& idx) const {
	size_t ijk = 0;
	for (size_t d = 0; d < ND; ++d)
	  ijk += strides[d]*idx[d];
	return ijk;
      }
	
      // Overloads for 1-3D
      __targetBoth__ size_t IndexToOffset(const size_t& i) const {
	static_assert(ND == 1, "Only valid for 1D arrays");
	return i * strides[0];
      }
      __targetBoth__ size_t IndexToOffset(const size_t i, const size_t j) const {
	static_assert(ND == 2, "Only valid for 2D arrays");
	return i * strides[0] + j * strides[1];
      }
      __targetBoth__ size_t IndexToOffset(const size_t i, const size_t j, const size_t k) const {
	static_assert(ND == 3, "Only valid for 3D arrays");
	return i * strides[0] + j * strides[1] + k * strides[2];
      }
	
      __targetBoth__ ShapeType OffsetToIndex(const size_t& ijk) {
	ShapeType idx;
	for (size_t d = 0; d < ND; ++d) {
	  idx[d] = ijk / strides[d];
	  ijk = ijk - idx[d] * strides[d];
	}
	return idx;
      }
	
      __targetBoth__ size_t Pitch() const {
	return element_pitch;
      }

      __targetBoth__ size_t MinStorageSize() const {
	const auto element_pitch_bytes = element_pitch * sizeof(T);
	return nElem * element_pitch_bytes;
      }
      
      __targetBoth__ size_t RawStorageSize() const {
	return MinStorageSize() + Alignment();
      }
      
      __targetBoth__ T* AlignRawStorage(void* raw_ptr) const {
	void* tmp = raw_ptr;
	const auto unpadded_buffer_size_bytes = MinStorageSize();
	auto space = RawStorageSize();
	auto ans = std::align(Alignment(), unpadded_buffer_size_bytes, tmp, space);
	assert(ans != nullptr);
	return static_cast<T*>(tmp);
      }
            
      template <class ArrayT, class IntT>	
      static void CreateBufferData(ArrayT& array, IntT*& shape, IntT*& strides, IntT*& suboffsets, void*& internal) {
	suboffsets = nullptr;
	auto& aShape = array.Shape();
	
	for (size_t i = 0; i < ND; ++i) {
	  shape[i] = aShape[i];
	  strides[i] = sizeof(ElemType) * array.indexer.strides[i];
	}
	shape[ND] = nElem;
	strides[ND] = sizeof(ElemType) * array.indexer.element_pitch;
	internal = nullptr;
      }
      
      static void ReleaseBufferData(void*& internal) {
	
      }	
      
      __targetBoth__ constexpr static size_t Alignment() {
	return ALIGN ? ALIGN : alignof(ElemType);
      }
      __targetBoth__ constexpr static size_t MaxVVL() {
	return MAX_VVL;
      }
    private:
      ShapeType strides;
      // Stride (in units of sizeof(T) bytes) between the elements of a contained item
      size_t element_pitch;
    };
  };

  
  
  // Core array class.
  
  // T = primitive element type
  // ND = number of dimensions
  // nElem = number of elements of type T making up an item
  // LayoutMetaFunc = class with a member template class named value that defines how to layout the data
  // The template must accept the arguments <T, ND, nElem> and define the following:
  //  - ShapeType - a type suitable for storing an ND index
  //  - any data needed to perform indexing
  //  - default constructable
  //  - constructable with a const ShapeType&
  //  - copyable & copy assignable
  //  - member funcs IndexToOffset & OffsetToIndex for converting
  //    between ShapeType and the offsets in the underlying array.
  //  - static member funcs CreateBufferData & ReleaseBufferData for
  //    populating the python buffer interface fields
  // Static methods:
  //  - Alignment
  //  - MaxVVL
  template <typename T, size_t ND, size_t nElem, class LayoutMetaFunc = SoALayoutPolicy<0,16> >
  class NdArray
  {
  public:
    // Typedefs
    typedef T ElemType;
    typedef typename LayoutMetaFunc::template value<T, ND, nElem> LayoutPolicy;
    typedef typename LayoutPolicy::ShapeType ShapeType;
    typedef ElemWrapper<T, nElem> WrapType;
    typedef ElemWrapper<const T, nElem> ConstWrapType;

    // Static property accessors
    __targetBoth__ constexpr static size_t Alignment();
    __targetBoth__ constexpr static size_t nElems();
    __targetBoth__ constexpr static size_t MaxVVL();
    __targetBoth__ constexpr static size_t nDims();
    
    // Default constructor
    __targetBoth__ NdArray();
    // Construct a given shape array
    __targetBoth__ NdArray(const ShapeType& shape);
    // Copy construct
    __targetBoth__ NdArray(const NdArray& other);
    // Copy assign
    __targetBoth__ NdArray& operator=(const NdArray& other);
    // Move construct
    __targetBoth__ NdArray(NdArray&& other);
    // Move assign
    __targetBoth__ NdArray& operator=(NdArray&& other);
    // Destruct
    __targetBoth__ ~NdArray();

    // Various property accessors
    __targetBoth__ bool OwnsData() const;
    __targetBoth__ const ShapeType& Shape() const;
    __targetBoth__ size_t nItems() const;
    __targetBoth__ T* Data();
    __targetBoth__ const T* Data() const;
    __targetBoth__ size_t DataSize() const;
    
    // Element access operators
    // [] - requires an ND index and returns a wrapped element
    // () - requires ND + 1 integer arguments to return a single value
    // Mutable and const overload of each are supplied
    
    __targetBoth__ WrapType operator[](const ShapeType& idx);
    __targetBoth__ ConstWrapType operator[](const ShapeType& idx) const;
  
    __targetBoth__ ElemType& operator()(const size_t i, const size_t d);
    __targetBoth__ const ElemType& operator()(const size_t i, const size_t d) const;

    __targetBoth__ ElemType& operator()(const size_t i, const size_t j, const size_t d);
    __targetBoth__ const ElemType& operator()(const size_t i, const size_t j, const size_t d) const;
  
    __targetBoth__ ElemType& operator()(const size_t i, const size_t j, const size_t k, const size_t d);
    __targetBoth__ const ElemType& operator()(const size_t i, const size_t j, const size_t k, const size_t d) const;

    // Get a ViewVL length slice of the array.
    // Only really meant to be used by target backends.
    template<size_t ViewVL>
    __targetBoth__ VectorView<NdArray, ViewVL> GetVectorView(size_t ijk);
    template<size_t ViewVL>
    __targetBoth__ VectorView<const NdArray, ViewVL> GetVectorView(size_t ijk) const;

    __targetBoth__ bool operator !=(const NdArray& other);

  private:
    // Helpers for move semantics
    
    // Set the state of this to the default-constructed state.
    __targetBoth__ void Reset();
    // Take copies of other's state into this
    __targetBoth__ void Steal(const NdArray& other);
    // Free resources held by this
    __targetBoth__ void Free();

    // Array shape
    ShapeType shape;
    // Total number of items in the array
    size_t size;
    
    // Indexing help - e.g. strides
    LayoutPolicy indexer;
    
    // Pointer to our zeroth element
    T* data;
    // Buffer size in units of sizeof(T)
    size_t buffer_size;
    
    // Underlying, un-aligned data storage & size
    // non-null => this instance owns the data & vice-versa
    void* raw_data;
    size_t raw_buffer_size_bytes;
    
    friend class SharedItem<NdArray>;
    // For some reason NVCC doesn't like using the typedef-ed name LayoutPolicy
    // friend LayoutPolicy;
    friend typename LayoutMetaFunc::template value<T, ND, nElem>;
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
