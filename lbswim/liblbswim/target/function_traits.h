// -*- mode: C++; -*-
#ifndef FUNCTION_TRAITS_H
#define FUNCTION_TRAITS_H

// For achieving a similar effect to :
// template<class... ParameterPack>
// struct example {
//   typedef ParameterPack Args;
// };
template<class... Args>
struct variadic_typedef {
};

// Return and argument types for functions
template<class FuncT>
struct function_traits {
};


// Main partial specialisation
template<class R, class... Args>
struct function_traits <R(Args...)> {
  typedef R(function_type)(Args...);
  typedef R return_type;
  typedef variadic_typedef<Args...> args_type;
};

#endif
