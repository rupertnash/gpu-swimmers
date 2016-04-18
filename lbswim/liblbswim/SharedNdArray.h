// -*- mode: C++; -*-
#ifndef SHAREDNDARRAY_H
#define SHAREDNDARRAY_H

#include "NdArray.h"
#include "SharedItem.h"

template<typename T, size_t ND, size_t nElem>
class SharedItem< NdArray<T,ND,nElem> >
{
 public:
  typedef NdArray<T, ND, nElem> SharedType;
  typedef typename SharedType::ShapeType ShapeType;
  
 private:
  SharedType* host;
  SharedType* device;
  T* device_data;
  char* raw_device_data;
  size_t dataSize;
  
  // Helpers for implementing move semantics
  void Reset();
  void Steal(const SharedItem&);
  void Free();
 public:
  SharedItem();
  SharedItem(const ShapeType& shape);

  // Disallow copy & allow move in ctor and assign
  SharedItem(const SharedItem&) = delete;
  SharedItem(SharedItem&& other);

  SharedItem& operator=(const SharedItem&) = delete;
  SharedItem& operator=(SharedItem&& other);
  
  ~SharedItem();

  SharedType& Host();
  SharedType* Device();
  
  const SharedType& Host() const;
  const SharedType* Device() const;
  void H2D();
  void D2H();

};

#endif
