find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "--std=c++11; -arch=compute_30; -code=compute_30"
  CACHE string "flags" FORCE)

cuda_add_library(lbswim SHARED lbswim.cu)

install(TARGETS lbswim DESTINATION ${_INST_DIR})
