// -*- mode: C++; -*-
#ifndef SHAREDITEM_HPP
#define SHAREDITEM_HPP

#include "SharedItem.h"
#include "targetpp.h"

template<typename T>
void SharedItem<T>::DevAlloc() {
  targetMalloc(device, sizeof(T));
}

template <typename T>
SharedItem<T>::~SharedItem() {
  delete host;
  targetFree(device);
}

template <typename T>
void SharedItem<T>::H2D() {
  copyToTarget(device,
	       host,
	       sizeof(T));
}

template <typename T>
void SharedItem<T>::D2H() {
  copyFromTarget(host,
		 device,
		 sizeof(T));
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
T& SharedItem<T>::Device() {
  return *device;
}
template <typename T>
const T& SharedItem<T>::Device() const {
  return *device;
}

#endif // SHAREDITEM_HPP
