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
  typedef typename ArrayType::LayoutPolicy LayoutPolicy;
  ArrayType& impl;
public:
  ArrayHelper(ArrayType& impl_) : impl(impl_)
  {}

  void GetBuffer(Py_buffer* view, int flags) {
    const ShapeType& shape = impl.Shape();
    //const ShapeType& strides = impl.Strides();
    const int nDims = impl.nDims();
    const int nElems = impl.nElems();
    
    view->buf = impl.Data();
    view->len = impl.DataSize() * sizeof(ElemType);
    view->readonly = std::is_const<ArrayType>::value;
    // This is OK because Python promises not to alter this string
    view->format = const_cast<char*>(TypeHelper<ElemType>::format);
    view->ndim = nDims+1;
    view->itemsize = sizeof(ElemType);
    view->shape = new Py_ssize_t[nDims + 1];
    view->strides = new Py_ssize_t[nDims + 1];
    LayoutPolicy::CreateBufferData(impl, view->shape, view->strides, view->suboffsets, view->internal);
  }

  void ReleaseBuffer(Py_buffer* view) {
    LayoutPolicy::ReleaseBufferData(view->internal);
  }
};

#endif // ARRAYHELPER_H
