// -*- mode: C++; -*-
#ifndef SHARED_H
#define SHARED_H

template<typename T>
struct SharedArray
{
  T* host;
  T* device;
  size_t size;
  SharedArray(size_t n);
  ~SharedArray();
  void H2D();
  void D2H();
};

template<typename T>
struct SharedItem {
  T* host;
  T* device;
  SharedItem();
  SharedItem(const T* init);
  ~SharedItem();
  void H2D();
  void D2H();
};


#endif
