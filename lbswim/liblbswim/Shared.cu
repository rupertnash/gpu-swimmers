#include "Shared.h"
#include "cucall.h"

template <typename T>
SharedArray<T>::SharedArray(size_t n) : size(n) {
  host = new T[n];
  CUDA_SAFE_CALL(cudaMalloc(&device, n * sizeof(T)));
}
template <typename T>
SharedArray<T>::~SharedArray() {
  delete[] host;
  CUDA_SAFE_CALL(cudaFree(device));
}
template <typename T>
void SharedArray<T>::H2D() {
  CUDA_SAFE_CALL(cudaMemcpy(device,
			    host,
			    size * sizeof(T),
			    cudaMemcpyHostToDevice));
}
template <typename T>
void SharedArray<T>::D2H() {
  CUDA_SAFE_CALL(cudaMemcpy(host,
			    device,
			    size * sizeof(T),
			    cudaMemcpyDeviceToHost));
}

template <typename T>
SharedItem<T>::SharedItem() {
  host = new T;
  CUDA_SAFE_CALL(cudaMalloc(&device, sizeof(T)));
}
template <typename T>
SharedItem<T>::SharedItem(const T* init) {
  host = new T;
  *host = *init;
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
