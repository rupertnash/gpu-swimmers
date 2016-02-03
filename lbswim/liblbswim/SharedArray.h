// -*- mode: C++; -*-
#ifndef SHAREDARRAY_H
#define SHAREDARRAY_H

#include "Array.h"
#include "SharedItem.h"

template<typename T, size_t ND, size_t nElem>
class SharedItem< Array<T,ND,nElem> > : public Array<T, ND, nElem>
{
 public:
  typedef Array<T, ND, nElem> Super;
 private:
  Super* device;
  T* dev_data;
 public:
  template<typename... Args>
  SharedItem(Args... args);
  ~SharedItem();

  Super* Host();
  Super* Device();
  
  const Super* Host() const;
  const Super* Device() const;
  void H2D();
  void D2H();

  static void SwapDevicePointers(SharedItem& a, SharedItem&b);
};

#endif
