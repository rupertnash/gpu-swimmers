find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -arch=sm_35)

cuda_add_library(lbswim SHARED lbswim.cu)

#add_executable(main main.cpp)
#target_link_libraries(main lbswim)

install(TARGETS lbswim DESTINATION ${_INST_DIR})
