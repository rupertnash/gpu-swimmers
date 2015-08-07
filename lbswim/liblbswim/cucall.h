#ifndef CUCALL_H
#define CUCALL_H
#include <iostream>

#define CUDA_SAFE_CALL(call) do {					\
    cudaError err = call;						\
    if (err != cudaSuccess) {						\
      std::cerr << "CUDA error in file '" << __FILE__			\
		<< "' on line " << __LINE__				\
		<< ". Message: " << cudaGetErrorString(err)		\
		<< std::endl;						\
      std::exit(EXIT_FAILURE);						\
    }									\
  } while(0)


#endif
