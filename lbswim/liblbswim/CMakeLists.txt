
include_directories(${TARGETDP_SOURCE_DIR})

if(TARGET_MODE STREQUAL "CUDA")
  find_package(CUDA REQUIRED)
  
  set(CUDA_NVCC_FLAGS "--std=c++11;-arch=compute_30;-code=compute_30"
    CACHE string "flags" FORCE)
  set(TARGET_DEFAULT_VVL 1
    CACHE STRING  "targetDP default virtual vector length")
  add_definitions("-DTARGET_DEFAULT_VVL=${TARGET_DEFAULT_VVL}")
  
  cuda_add_library(lbswim SHARED lbswim.cu)

elseif((TARGET_MODE STREQUAL "OPENMP") OR (TARGET_MODE STREQUAL "VANILLA"))

  if(TARGET_MODE STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_CXX_FLAGS})
  endif()
  set(TARGET_DEFAULT_VVL 8
    CACHE STRING  "targetDP default virtual vector length")
  add_definitions("-DTARGET_DEFAULT_VVL=${TARGET_DEFAULT_VVL}")

  add_library(lbswim SHARED
    Lattice.cxx  SwimmerArray.cxx TracerArray.cxx python_template_instantiations.cxx
    target/rand_cpp.cxx ${TARGETDP_SOURCE_DIR}/targetDP_C.c
    )
  target_link_libraries(lbswim ${OpenMP_CXX_FLAGS})
  
  set_source_files_properties(${TARGETDP_SOURCE_DIR}/targetDP_C.c
    PROPERTIES COMPILE_FLAGS "-x c++")

  add_executable(lb_test lb_test.cxx)
  target_link_libraries(lb_test lbswim)
else()
  message(FATAL_ERROR "Unknown TARGET_MODE = '${TARGET_MODE}'")
endif()

install(TARGETS lbswim DESTINATION ${_INST_DIR})
