# Defines the CMake commands/policies
cmake_minimum_required( VERSION 2.8.5 )

# Set the project name
project(LBSWIMMERS)

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

find_package(PythonInterp)
find_package(PythonLibs)

execute_process( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

# If not specified, guess where to install the python
set(LBSWIMMERS_PACKAGE_INSTALL_DIR ${PYTHON_SITE_PACKAGES}
  CACHE PATH "Directory where the package will get installed - should be on your PYTHONPATH")
set(_INST_DIR "${LBSWIMMERS_PACKAGE_INSTALL_DIR}/lbswim")

# Consider addinng a check that LBSWIMMERS_PACKAGE_INSTALL_DIR is indeed in sys.path


# Sort out RPATH issues
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${_INST_DIR}")

add_subdirectory( lbswim )
