#ifndef TARGET_NDARRAYIMPL_HPP
#define TARGET_NDARRAYIMPL_HPP

namespace target {
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Alignment() {
    return ALIGN ? ALIGN : alignof(T);
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ bool NdArray<T, ND, nElem, ALIGN, MAX_VVL>::OwnsData() const {
    return raw_data != nullptr;
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::nElems() {
    return nElem;
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::MaxVVL() {
    return MAX_VVL;
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::nDims() {
    return ND;
  }
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Shape() const -> const ShapeType& {
    return indexer.shape;
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Strides() const -> const ShapeType& {
    return indexer.strides;
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Size() const {
    return indexer.size;
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ size_t NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Pitch() const {
    return element_pitch;
  }
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ T* NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Data() {
    return data;
  }
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ const T* NdArray<T, ND, nElem, ALIGN, MAX_VVL>::Data() const {
    return data;
  }
  
  // Default constructor
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ NdArray<T, ND, nElem, ALIGN, MAX_VVL>::NdArray()
    : indexer(ShapeType()), data(nullptr), buffer_size_bytes(0), raw_data(nullptr), element_pitch(0)
  {
  }
  
  // Construct a given shape array
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ NdArray<T, ND, nElem, ALIGN, MAX_VVL>::NdArray(const ShapeType& shape)
    : indexer(shape)
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

  // Copy constructor
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ NdArray<T, ND, nElem, ALIGN, MAX_VVL>::NdArray(const NdArray& other)
    : indexer(other.indexer),
      data(other.data),
      buffer_size_bytes(other.buffer_size_bytes),
      raw_data(nullptr),
      element_pitch(other.element_pitch)
  {
  }

  // copy assign
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator=(const NdArray& other) -> NdArray& {
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

  // Move ctor
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ NdArray<T, ND, nElem, ALIGN, MAX_VVL>::NdArray(NdArray&& other)
    : indexer(other.indexer),
      data(other.data),
      buffer_size_bytes(other.buffer_size_bytes),
      raw_data(other.raw_data),
      element_pitch(other.element_pitch)
  {
    other.indexer = IdxType();
    other.data = nullptr;
    other.raw_data = nullptr;
  }

  // Move assign
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator=(NdArray&& other) -> NdArray& {
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

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ NdArray<T, ND, nElem, ALIGN, MAX_VVL>::~NdArray() {
    if (OwnsData()) {
      std::free(raw_data);
      raw_data = nullptr;
      data = nullptr;
    }
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator[](const ShapeType& idx) -> WrapType {
    WrapType ans;
    ans.data = data + indexer.nToOne(idx);
    // Refactor to allow layout switching
    ans.stride = element_pitch;
    return ans;
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator[](const ShapeType& idx) const -> ConstWrapType {
    ConstWrapType ans;
    ans.data = data + indexer.nToOne(idx);
    // Refactor to allow layout switching
    ans.stride = element_pitch;
    return ans;
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t d) -> ElemType& {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[i*indexer.strides[0] + element_pitch*d];
  }
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[i*indexer.strides[0] + element_pitch*d];
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t j, const size_t d) ->  ElemType&  {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + element_pitch*d];
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t j, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + element_pitch*d];
  }

  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t j, const size_t k, const size_t d) ->  ElemType&  {
    static_assert(nDims() == 3, "Only valid for 3D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + element_pitch*d];
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator()(const size_t i, const size_t j, const size_t k, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 3, "Only valid for 3D arrays");
    return data[i * indexer.strides[0] + j * indexer.strides[1] + k * indexer.strides[2] + element_pitch*d];
  }
  
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  template<size_t ViewVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::GetVectorView(size_t ijk) -> VectorView<NdArray, ViewVL> {
    VectorView<NdArray, ViewVL> ans;
    ans.zero.data = data + ijk;
    ans.zero.stride = element_pitch;
    return ans;
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  template<size_t ViewVL>
  __targetBoth__ auto NdArray<T, ND, nElem, ALIGN, MAX_VVL>::GetVectorView(size_t ijk) const -> VectorView<const NdArray, ViewVL> {
    VectorView<const NdArray, ViewVL> ans;
    ans.zero.data = data + ijk;
    ans.zero.stride = element_pitch;
    return ans;
  }
  template <typename T, size_t ND, size_t nElem, size_t ALIGN, size_t MAX_VVL>
  __targetBoth__ bool NdArray<T, ND, nElem, ALIGN, MAX_VVL>::operator!=(const NdArray& other) {
    return data != other.data;
  }
  
}

#endif
