// Definitions of the templates

#ifndef TARGET_TARGETPP_HPP
#define TARGET_TARGETPP_HPP

#include "./target.h"

namespace target {
  template<typename T>
  void malloc(T*& ptr, const size_t n) {
    targetMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n);
  }
  
  template<typename T>
  void free(T* ptr) {
    targetFree(reinterpret_cast<void*>(ptr));
  }
  
  template<typename T>
  void copyIn(T* targetData, const T* data, const size_t n) {
    copyToTarget(targetData, data, sizeof(T) * n);
  }

  template<typename T>
  void copyOut(T* data, const T* targetData, const size_t n) {
    copyFromTarget(data, targetData, sizeof(T) * n);
  }

}

#endif
