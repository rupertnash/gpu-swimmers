#ifndef TARGET_NDARRAYIMPL_HPP
#define TARGET_NDARRAYIMPL_HPP

namespace target {
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, LayoutT>::Alignment() {
    return LayoutPolicy::Alignment();
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ bool NdArray<T, ND, nElem, LayoutT>::OwnsData() const {
    return raw_data != nullptr;
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, LayoutT>::nElems() {
    return nElem;
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, LayoutT>::MaxVVL() {
    return LayoutPolicy::MaxVVL();
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ constexpr size_t NdArray<T, ND, nElem, LayoutT>::nDims() {
    return ND;
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::Shape() const -> const ShapeType& {
    return shape;
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ size_t NdArray<T, ND, nElem, LayoutT>::nItems() const {
    return size;
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ T* NdArray<T, ND, nElem, LayoutT>::Data() {
    return data;
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ const T* NdArray<T, ND, nElem, LayoutT>::Data() const {
    return data;
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ size_t NdArray<T, ND, nElem, LayoutT>::DataSize() const {
    return buffer_size;
  }
  
  // Set the state of this to the default-constructed state.
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ void NdArray<T, ND, nElem, LayoutT>::Reset() {
    shape = ShapeType();
    size = 0;
    indexer = LayoutPolicy();
    buffer_size = 0;
    data = nullptr;
    raw_buffer_size_bytes = 0;
    raw_data = nullptr;
  }
  
  // Take copies of other's state into this
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ void NdArray<T, ND, nElem, LayoutT>::Steal(const NdArray& other) {
    shape = other.shape;
    size = other.size;
    indexer = other.indexer;
    buffer_size = other.buffer_size;
    data = other.data;
    raw_buffer_size_bytes = other.raw_buffer_size_bytes;
    raw_data = other.raw_data;
  }
  
  // Free resources held by this
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ void NdArray<T, ND, nElem, LayoutT>::Free() {
    if (OwnsData()) {
      std::free(raw_data);
      raw_data = nullptr;
    }
  }

  // Default constructor
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ NdArray<T, ND, nElem, LayoutT>::NdArray() {
    Reset();
  }
  
  // Construct a given shape array
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ NdArray<T, ND, nElem, LayoutT>::NdArray(const ShapeType& shape)
    : shape(shape), size(Product(shape)), indexer(shape)
  {
    raw_buffer_size_bytes = indexer.RawStorageSize();
    raw_data = std::malloc(raw_buffer_size_bytes);
    assert(raw_data != nullptr);
    data = indexer.AlignRawStorage(raw_data);
    
    auto buffer_size_bytes = indexer.MinStorageSize();
    buffer_size = buffer_size_bytes / sizeof(T);
  }

  // Copy constructor
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ NdArray<T, ND, nElem, LayoutT>::NdArray(const NdArray& other)
    : indexer(other.indexer),
      data(other.data),
      raw_buffer_size_bytes(other.raw_buffer_size_bytes),
      // Note that raw = null => this doesn't own the data
      raw_data(nullptr)
  {
  }

  // copy assign
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator=(const NdArray& other) -> NdArray& {
    Free();
    // Note that raw = null => this doesn't own the data    
    indexer = other.indexer;
    data = other.data;
    raw_buffer_size_bytes = other.raw_buffer_size_bytes;
    return *this;
  }

  // Move ctor
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ NdArray<T, ND, nElem, LayoutT>::NdArray(NdArray&& other)
  {
    Steal(other);
    other.Reset();
  }

  // Move assign
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator=(NdArray&& other) -> NdArray& {
    Free();
    Steal(other);
    other.Reset();
    return *this;
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ NdArray<T, ND, nElem, LayoutT>::~NdArray() {
    Free();
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator[](const ShapeType& idx) -> WrapType {
    WrapType ans;
    ans.data = data + indexer.IndexToOffset(idx);
    // Refactor to allow layout switching
    ans.stride = indexer.Pitch();
    return ans;
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator[](const ShapeType& idx) const -> ConstWrapType {
    ConstWrapType ans;
    ans.data = data + indexer.IndexToOffset(idx);
    // Refactor to allow layout switching
    ans.stride = indexer.Pitch();
    return ans;
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t d) -> ElemType& {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[indexer.IndexToOffset(i) + indexer.Pitch()*d];
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 1, "Only valid for 1D arrays");
    return data[indexer.IndexToOffset(i) + indexer.Pitch()*d];
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t j, const size_t d) ->  ElemType&  {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[indexer.IndexToOffset(i, j) + indexer.Pitch()*d];
  }
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t j, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 2, "Only valid for 2D arrays");
    return data[indexer.IndexToOffset(i, j) + indexer.Pitch()*d];
  }

  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t j, const size_t k, const size_t d) ->  ElemType&  {
    static_assert(nDims() == 3, "Only valid for 3D arrays");
    return data[indexer.IndexToOffset(i, j, k) + indexer.Pitch()*d];
  }
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::operator()(const size_t i, const size_t j, const size_t k, const size_t d) const -> const ElemType& {
    static_assert(nDims() == 3, "Only valid for 3D arrays");
    return data[indexer.IndexToOffset(i, j, k) + indexer.Pitch()*d];
  }
  
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  template<size_t ViewVL>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::GetVectorView(size_t ijk) -> VectorView<NdArray, ViewVL> {
    VectorView<NdArray, ViewVL> ans;
    ans.zero.data = data + ijk;
    ans.zero.stride = indexer.Pitch();
    return ans;
  }
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  template<size_t ViewVL>
  __targetBoth__ auto NdArray<T, ND, nElem, LayoutT>::GetVectorView(size_t ijk) const -> VectorView<const NdArray, ViewVL> {
    VectorView<const NdArray, ViewVL> ans;
    ans.zero.data = data + ijk;
    ans.zero.stride = indexer.Pitch();
    return ans;
  }
  template <typename T, size_t ND, size_t nElem, typename LayoutT>
  __targetBoth__ bool NdArray<T, ND, nElem, LayoutT>::operator!=(const NdArray& other) {
    return data != other.data;
  }
  
}

#endif
