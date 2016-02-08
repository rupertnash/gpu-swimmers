// -*- mode: C++; -*-
#ifndef SHAREDARRAY_H
#define SHAREDARRAY_H

#include "Array.h"
#include "SharedItem.h"

template<typename T, size_t ND, size_t nElem>
class SharedItem< Array<T,ND,nElem> >
{
 public:
  typedef Array<T, ND, nElem> SharedType;
  typedef typename SharedType::ShapeType ShapeType;
  
 private:
  SharedType* host;
  SharedType* device;
  T* device_data;
  size_t dataSize;
  
  void Reset();
  void Steal(const SharedItem&);
  void Free();
 public:
  SharedItem();
  SharedItem(const ShapeType& shape);

  // Disallow copy &  allow move in ctor and assign
  SharedItem(const SharedItem&) = delete;
  SharedItem(SharedItem&& other);

  SharedItem& operator=(const SharedItem&) = delete;
  SharedItem& operator=(SharedItem&& other);
  
  ~SharedItem();

  SharedType& Host();
  SharedType& Device();
  
  const SharedType& Host() const;
  const SharedType& Device() const;
  void H2D();
  void D2H();

};

#endif
