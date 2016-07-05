// -*- mode: C++; -*-

#ifndef ARRAYHELPER_H
#define ARRAYHELPER_H
#include <type_traits>

template<typename T>
struct TypeHelper {
  static const char* format;
};
template<> const char* TypeHelper<double>::format = "d";

template<typename ArrayType>
class ArrayHelper {
  typedef typename ArrayType::ShapeType ShapeType;
  typedef typename ArrayType::ElemType ElemType;

  ArrayType& impl;
public:
  ArrayHelper(ArrayType& impl_) : impl(impl_)
  {}

  void GetBuffer(Py_buffer* view, int flags) {
    const ShapeType& shape = impl.Shape();
    const ShapeType& strides = impl.Strides();
    const int nDims = impl.nDims();
    const int nElems = impl.nElems();
    
    view->buf = impl.Data();
    view->len = impl.Size() * nElems * sizeof(ElemType);
    view->readonly = std::is_const<ArrayType>::value;
    // This is OK because Python promises not to alter this string
    view->format = const_cast<char*>(TypeHelper<ElemType>::format);
    view->ndim = nDims+1;
    view->itemsize = sizeof(ElemType);

    view->shape = new Py_ssize_t[nDims + 1];
    view->strides = new Py_ssize_t[nDims + 1];
    view->suboffsets = NULL;
    for (size_t i = 0; i < nDims; ++i) {
      view->shape[i] = shape[i];
      view->strides[i] = sizeof(ElemType) * strides[i];
    }
    view->shape[nDims] = nElems;
    // Needs factoring to allow layout switching
    view->strides[nDims] = sizeof(ElemType) * impl.Pitch();

    view->internal = NULL;
  }

  void ReleaseBuffer(Py_buffer* view) {
    delete[] view->shape;
    delete[] view->strides;
  }
};

#endif // ARRAYHELPER_H
