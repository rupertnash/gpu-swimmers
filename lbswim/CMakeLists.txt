# Do the CUDA library first
add_subdirectory( liblbswim )

include(UseCython)
find_package(NumPy)

# Get the list of .pyx files
FILE(GLOB PYX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} *.pyx)
# Set them as cython c++
set_source_files_properties(${PYX_SOURCES}
  PROPERTIES CYTHON_IS_CXX TRUE)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/liblbswim ${NUMPY_INCLUDE_DIRS})

foreach(basename shared lb swimmers tracers)
  cython_add_module(${basename} ${basename}.pyx)
  target_link_libraries(${basename} lbswim)
  install(TARGETS ${basename}
    DESTINATION ${_INST_DIR})
endforeach(basename)

foreach(py_mod __init__ System)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${py_mod}.py
    DESTINATION ${_INST_DIR})
endforeach()