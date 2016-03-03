
include_directories(${TARGETDP_SOURCE_DIR})

if(TARGET_MODE STREQUAL "CUDA")
  find_package(CUDA REQUIRED)
  
  set(CUDA_NVCC_FLAGS "--std=c++11;-arch=compute_30;-code=compute_30"
    CACHE string "flags" FORCE)
  
  cuda_add_library(lbswim SHARED lbswim.cu)
elseif((TARGET_MODE STREQUAL "OPENMP") OR (TARGET_MODE STREQUAL "VANILLA"))

  if(TARGET_MODE STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_CXX_FLAGS})
  endif()
  
  add_library(lbswim SHARED
    Lattice.cxx  SwimmerArray.cxx TracerArray.cxx python_template_instantiations.cxx
    target/rand_cpp.cxx ${TARGETDP_SOURCE_DIR}/targetDP_C.c
    )
  
  set_source_files_properties(${TARGETDP_SOURCE_DIR}/targetDP_C.c
    PROPERTIES COMPILE_FLAGS "-x c++")
  
else()
  message(FATAL_ERROR "Unknown TARGET_MODE = '${TARGET_MODE}'")
endif()

install(TARGETS lbswim DESTINATION ${_INST_DIR})
