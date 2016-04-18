// -*- mode: C++; -*-
#ifndef TARGET_CPP_BACKEND_H
#define TARGET_CPP_BACKEND_H

#include "./func_attr.h"
#include "../array.h"
#include "../Nd.h"
#include "./function_traits.h"

namespace target {
  
  // Forward declarations
  template<size_t, size_t>
  struct CppThreadContext;
  template<class, size_t>
  struct VectorView;
  
  // Target-only class that controls the iteration over an index space
  template<size_t ND = 1, size_t VL = VVL>
  struct CppContext {
  
    typedef array<size_t, ND> Shape;
  
    // Constructors
    __target__ CppContext(const Shape& n);
    __target__ CppContext(const size_t n);
  
    // Container protocol
    __target__ CppThreadContext<ND, VL> begin() const;
    __target__ CppThreadContext<ND, VL> end() const;
  
    // Describe the index space the global iteration is over
    const Shape start;
    const Shape extent;
    const SpaceIndexer<ND> indexer;
  };
  template<size_t ND = 1, size_t VL = VVL>
  bool operator==(const CppContext<ND, VL>& a, const CppContext<ND, VL>& b);
  
  // Target-only class that controls thread-level iteration over the
  // index space.
  
  // OpenMP requires that it be a random access iterator.
  template<size_t ND = 1, size_t VL = VVL>
  struct CppThreadContext /*: public std::iterator<std::random_access_iterator_tag,
						 const array<size_t, ND>,
						 int>*/
  {
  
    typedef CppContext<ND, VL> Context;
    typedef typename Context::Shape Shape;
  
    // Constructor
    __target__ CppThreadContext() = default;
    __target__ CppThreadContext(const Context& ctx_, const Shape& pos);
    __target__ CppThreadContext(const CppThreadContext& other) = default;
    __target__ CppThreadContext(CppThreadContext&& other) = default;

    // Destructor
    __target__ ~CppThreadContext() = default;
    
    // Assign
    __target__ CppThreadContext& operator=(const CppThreadContext& other) = default;
    __target__ CppThreadContext& operator=(CppThreadContext&& other) = default;
    
    // Deref
    __target__ size_t operator*() const;
    // Increment/decrement
    __target__ CppThreadContext& operator++();
    __target__ CppThreadContext& operator--();
    __target__ CppThreadContext& operator+=(std::ptrdiff_t);
    __target__ CppThreadContext& operator-=(std::ptrdiff_t);

    __target__ size_t operator[](std::ptrdiff_t) const;

    // Get the current index slice from an array
    template <class ArrayT>
    __target__ VectorView<ArrayT, VL> GetCurrentElements(ArrayT* arr);
    
    Shape GetNdIndex(size_t i) const;
    
    __target__ constexpr size_t VectorLength() const {
      return VL;
    }
    // Global iteration space
    const Context& ctx;
    // Current position of this thread in its iteration
    size_t ijk;
  };
  
  template<size_t ND = 1, size_t VL = VVL>
  bool operator==(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  template<size_t ND = 1, size_t VL = VVL>
  bool operator!=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  template<size_t ND = 1, size_t VL = VVL>
  bool operator<(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  template<size_t ND = 1, size_t VL = VVL>
  bool operator<=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  template<size_t ND = 1, size_t VL = VVL>
  bool operator>(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  template<size_t ND = 1, size_t VL = VVL>
  bool operator>=(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);

  template<size_t ND = 1, size_t VL = VVL>
  std::ptrdiff_t operator-(const CppThreadContext<ND, VL>& a, const CppThreadContext<ND, VL>& b);
  
  // // Target-only class for instruction-level iteration over the space
  template<class AT, size_t VL = VVL>
  struct VectorView {
    typedef AT ArrayT;
    typedef typename AT::ElemType ElemType;
    typedef typename AT::WrapType WrapType;

    WrapType zero;
    WrapType operator[](size_t i);
  };
  

  template<class Impl>
  struct Kernel {
    template<size_t ND>
    struct Dims {
      typedef array<size_t, ND> Index;
      template<size_t VL>
      struct VecLen {
	template<class... Args>
	struct ArgTypes {
	  Index extent;
	  CppContext<ND,VL>* indexSpace;
	  __targetBoth__ ArgTypes(const Index& shape);
	  void operator()(Args... args);
	  constexpr size_t Dims() const {
	    return ND;
	  }
	  constexpr size_t VecLen() const {
	    return VL;
	  }
	  
	};
      };
    };
  };
  
}

#endif
