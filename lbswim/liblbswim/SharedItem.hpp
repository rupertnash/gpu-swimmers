// -*- mode: C++; -*-
#ifndef SHAREDITEM_HPP
#define SHAREDITEM_HPP

#include "SharedItem.h"
#include "cucall.h"

template <typename T>
SharedItem<T>::SharedItem() {
  host = new T;
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
template<typename A1>
SharedItem<T>::SharedItem(A1 a1) {
  host = new T(a1);
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
template<typename A1, typename A2>
SharedItem<T>::SharedItem(A1 a1, A2 a2) {
  host = new T(a1, a2);
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
template<typename A1, typename A2, typename A3>
SharedItem<T>::SharedItem(A1 a1, A2 a2, A3 a3) {
  host = new T(a1, a2, a3);
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
template<typename A1, typename A2, typename A3, typename A4>
SharedItem<T>::SharedItem(A1 a1, A2 a2, A3 a3, A4 a4) {
  host = new T(a1, a2, a3, a4);
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
SharedItem<T>::SharedItem(const T* init) {
  host = new T;
  *host = *init;
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
SharedItem<T>::SharedItem(const T& init) {
  host = new T;
  *host = init;
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
  H2D();
}

template <typename T>
SharedItem<T>::~SharedItem() {
  delete host;
  CUDA_SAFE_CALL(cudaFree(device));
}
template <typename T>
void SharedItem<T>::H2D() {
  CUDA_SAFE_CALL(cudaMemcpy(device,
			    host,
			    sizeof(T),
			    cudaMemcpyHostToDevice));
}
template <typename T>
void SharedItem<T>::D2H() {
  CUDA_SAFE_CALL(cudaMemcpy(host,
			    device,
			    sizeof(T),
			    cudaMemcpyDeviceToHost));
}

template <typename T>
T* SharedItem<T>::Host() {
  return host;
}
template <typename T>
const T* SharedItem<T>::Host() const {
  return host;
}

template <typename T>
T* SharedItem<T>::Device() {
  return device;
}
template <typename T>
const T* SharedItem<T>::Device() const {
  return device;
}
  

template <typename T>
T& SharedItem<T>::operator*() {
#ifdef __CUDA_ARCH__
  return *device;
#else
  return *host;
#endif
}

template <typename T>
const T& SharedItem<T>::operator*() const {
#ifdef __CUDA_ARCH__
  return *device;
#else
  return *host;
#endif
}

template <typename T>
T* SharedItem<T>::operator->() {
#ifdef __CUDA_ARCH__
  return device;
#else
  return host;
#endif
}

template <typename T>
const T* SharedItem<T>::operator->() const {
#ifdef __CUDA_ARCH__
  return device;
#else
  return host;
#endif
}

#endif // SHAREDITEM_HPP
