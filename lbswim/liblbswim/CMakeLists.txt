
include_directories(${TARGETDP_SOURCE_DIR})

if(TARGET_MODE STREQUAL "CUDA")
  find_package(CUDA REQUIRED)
  
  set(CUDA_NVCC_FLAGS "--std=c++11;-arch=compute_30;-code=compute_30"
    CACHE string "flags" FORCE)
  
  cuda_add_library(lbswim SHARED lbswim.cu)
elseif(TARGET_MODE STREQUAL "OPENMP")
  message(FATAL_ERROR "TARGET_MODE = OPENMP not implemented")

elseif(TARGET_MODE STREQUAL "VANILLA")
  add_library(lbswim SHARED
    Lattice.cxx  SwimmerArray.cxx TracerArray.cxx python_template_instantiations.cxx
    target/rand_cpp.cxx ${TARGETDP_SOURCE_DIR}/targetDP_C.c
    )
else()
  message(FATAL_ERROR "Unknown TARGET_MODE = '${TARGET_MODE}'")
endif()

install(TARGETS lbswim DESTINATION ${_INST_DIR})
