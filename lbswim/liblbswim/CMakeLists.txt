find_package(CUDA REQUIRED)
find_package(TargetDP REQUIRED)

include_directories(${TARGETDP_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "--std=c++11; -arch=compute_30; -code=compute_30"
  CACHE string "flags" FORCE)

cuda_add_library(lbswim SHARED lbswim.cu)

target_link_libraries(lbswim ${TARGETDP_LIBRARIES})

install(TARGETS lbswim DESTINATION ${_INST_DIR})
