# Do the CUDA library first

set(TARGETDP_SOURCE_DIR ""
  CACHE path "Path to the targetDP source")
set(TARGET_MODE "CUDA"
  CACHE string "CUDA, OPENMP or VANILLA")

add_definitions(-DTARGET_MODE=${TARGET_MODE} -DTARGET_MODE_${TARGET_MODE})
add_compile_options(--std=c++11)

add_subdirectory( liblbswim )

include(UseCython)
find_package(NumPy)

# Get the list of .pyx files
FILE(GLOB PYX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} *.pyx)
# Set them as cython c++
set_source_files_properties(${PYX_SOURCES}
  PROPERTIES CYTHON_IS_CXX TRUE)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/liblbswim ${NUMPY_INCLUDE_DIRS} ${TARGETDP_SOURCE_DIR})

foreach(basename Array Fields Lists lb swimmers tracers)
  cython_add_module(${basename} ${basename}.pyx)
  target_link_libraries(${basename} lbswim)
  install(TARGETS ${basename}
    DESTINATION ${_INST_DIR})
endforeach(basename)

foreach(py_mod __init__ System)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${py_mod}.py
    DESTINATION ${_INST_DIR})
endforeach()