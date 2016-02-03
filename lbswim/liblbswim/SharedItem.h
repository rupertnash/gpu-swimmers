// -*- mode: C++; -*-
#ifndef SHAREDITEM_H
#define SHAREDITEM_H

template<typename T>
class SharedItem {
  T* host;
  T* device;
public:
  // Constructors taking zero to four parameters
  SharedItem();
  template<typename A1>
  SharedItem(A1 a1);
  template<typename A1, typename A2>
  SharedItem(A1 a1, A2 a2);
  template<typename A1, typename A2, typename A3>
  SharedItem(A1 a1, A2 a2, A3 a3);
  template<typename A1, typename A2, typename A3, typename A4>
  SharedItem(A1 a1, A2 a2, A3 a3, A4 a4);
  
  // Copy item constructors
  SharedItem(const T* init);
  SharedItem(const T& init);
  
  ~SharedItem();

  void H2D();
  void D2H();

  T* Host();
  const T* Host() const;

  T* Device();
  const T* Device() const;
  
  T* operator->();
  const T* operator->() const;

  T& operator*();
  const T& operator*() const;
};

#endif
