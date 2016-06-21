// -*- mode: C++; -*-
#ifndef TARGET_SHAREDITEM_HPP
#define TARGET_SHAREDITEM_HPP

#include "SharedItem.h"
#include "targetpp.h"

namespace target {
template<typename T>
void SharedItem<T>::DevAlloc() {
  target::malloc(device);
}

template <typename T>
SharedItem<T>::~SharedItem() {
  delete host;
  target::free(device);
}

template <typename T>
void SharedItem<T>::H2D() {
  target::copyIn(device,
		 host);
}

template <typename T>
void SharedItem<T>::D2H() {
  target::copyOut(host,
		  device);
}

template <typename T>
T& SharedItem<T>::Host() {
  return *host;
}
template <typename T>
const T& SharedItem<T>::Host() const {
  return *host;
}

template <typename T>
T* SharedItem<T>::Device() {
  return device;
}
template <typename T>
const T* SharedItem<T>::Device() const {
  return device;
}
}
#endif // SHAREDITEM_HPP
