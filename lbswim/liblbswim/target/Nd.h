// -*- mode: C++; -*-
#ifndef TARGET_ND_H
#define TARGET_ND_H

#include "func_attr.h"

#include "array.h"

namespace target {
  
  template <size_t ND>
  struct AoSSpaceLayoutPolicy
  {
    typedef array<size_t, ND> ShapeType;
    __targetBoth__ static ShapeType MakeStrides(const ShapeType& shape)
    {
      ShapeType strides;
      strides[ND-1] = 1;
      for (int i = int(ND) - 2; i >= 0; --i)
	strides[i] = strides[i+1] * shape[i+1];
      return strides;
    }
  };
  /*
    template <size_t ND>
    struct SoASpaceLayoutPolicy
    {
    typedef std::array<size_t, ND> ShapeType;
    __targetBoth__ static ShapeType MakeStrides(const ShapeType& shape)
    {
    ShapeType strides;
    strides[ND-1] = 1;
    for (int i = int(ND) - 2; i >= 0; --i)
    strides[i] = strides[i+1] * shape[i+1];
    return strides;
    }
    };
  */

  template <size_t ND>
  using SpaceLayoutPolicy = AoSSpaceLayoutPolicy<ND>;

  template <class T, size_t ND>
  __targetBoth__ T Product(const array<T,ND>& arr) {
    T ans = 1;
    for (size_t i = 0; i < ND; ++i) 
      ans *= arr[i];
    return ans;
  }

  // Helper template class for indexing.
  // ND = number of space dimensions.
  template <size_t ND>
  struct SpaceIndexer
  {
    static const size_t ndims;
    typedef array<size_t, ND> ShapeType;
    ShapeType shape;
    ShapeType strides;
    size_t size;
  
    __targetBoth__ SpaceIndexer() : shape(), strides(), size(0)
    {
    }
  
    // Construct for a given shape array.
    __targetBoth__ SpaceIndexer(const ShapeType& shp) : shape(shp),
							strides(SpaceLayoutPolicy<ND>::MakeStrides(shp)),
							size(Product(shp))
    {
    }
  
    // Compute 1D index from ND index
    __targetBoth__ size_t nToOne(const ShapeType& idx) const {
      size_t ijk = 0;
      for (size_t d = 0; d < ND; ++d)
	ijk += strides[d]*idx[d];
      return ijk;
    }
    // Compute ND index from 1D index
    __targetBoth__ ShapeType oneToN(size_t ijk) const {
      ShapeType idx;
      for (size_t d = 0; d < ND; ++d) {
	idx[d] = ijk / strides[d];
	ijk = ijk - idx[d] * strides[d];
      }
      return idx;
    }
  
    // Helper for subscripting - given a parent array of the next higher
    // dimensionality return an indexer for an array slice taken along
    // the first dimension.
    __targetBoth__ static SpaceIndexer ReduceFrom(const SpaceIndexer<ND+1>& parent) {
      SpaceIndexer ans;
      for (size_t i=0; i<ndims; ++i) {
	ans.shape[i] = parent.shape[i+1];
	ans.strides[i] = parent.strides[i+1];
      }
      ans.size = parent.size;
      return ans;
    }
  };

  // Define static member
  template<size_t ND>
  const size_t SpaceIndexer<ND>::ndims = ND;
}
#endif // ND_H
