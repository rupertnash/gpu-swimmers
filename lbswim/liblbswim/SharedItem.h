// -*- mode: C++; -*-
#ifndef SHAREDITEM_H
#define SHAREDITEM_H

template<typename T>
class SharedItem {
  T* host;
  T* device;
public:
  // Varadic construct that delegates to the shared type.
  template<typename... Args>
  SharedItem(Args... args);
  
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
