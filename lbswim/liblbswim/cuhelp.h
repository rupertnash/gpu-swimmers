// -*- mode: C++; -*-
#ifndef CUHELP_H
#define CUHELP_H

#ifdef __CUDACC__
// CUDA mode

#define HOST __host__
#define DEVICE __device__
#define BOTH HOST DEVICE
#define KERNEL __global__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename T>
using host_vector = thrust::host_vector<T>;
template<typename T>
using device_vector = thrust::device_vector<T>;

#else
// No CUDA

#define HOST
#define DEVICE
#define BOTH HOST DEVICE
#define KERNEL

#include <vector>

template<typename T>
using host_vector = std::vector<T>;
template<typename T>
using device_vector = std::vector<T>;

#endif

#endif // CUHELP_H
