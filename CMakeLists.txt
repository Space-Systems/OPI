# Minimum version required is 3.1 for CMAKE_CXX_STANDARD
cmake_minimum_required (VERSION 3.1)

# disable in source builds (these must be placed before project)
if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed")
endif()
# State the name of the project/executable and the programming language
project(OPI CXX)
option(ENABLE_FORTRAN_SUPPORT "Enable Fortran (default=ON)" ON)

enable_language(C OPTIONAL)
if(ENABLE_FORTRAN_SUPPORT)
  enable_language(Fortran OPTIONAL)
endif()

# add module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

# find cuda libraries
option(ENABLE_CUDA_SUPPORT "Enable cuda_support" ON)
if(ENABLE_CUDA_SUPPORT)
  if(WIN32)
	set(CUDA_64_BIT_DEVICE_CODE OFF)
  endif()
  find_package( CUDA )
  if( CUDA_FOUND )
    message("CUDA Support enabled")
  else()
    message("CUDA not found - Support disabled")
    set(ENABLE_CUDA_SUPPORT OFF)
  endif()
endif()
if(NOT ENABLE_CUDA_SUPPORT)
  add_definitions( -DOPI_DISABLE_CUDA )
endif()

# find OpenCL
option(ENABLE_CL_SUPPORT "Enable opencl_support" ON)
if(ENABLE_CL_SUPPORT)
  find_package( OpenCL )
  if( OpenCL_FOUND )
    message("OpenCL Support enabled")
  else()
    message("OpenCL not found - Support disabled")
    set(ENABLE_CL_SUPPORT OFF)
  endif()
endif()
if(NOT ENABLE_CL_SUPPORT)
    add_definitions( -DOPI_DISABLE_OPENCL )
endif()

option(ENABLE_PYTHON "Generate Python bindings using SWIG" ON)
if (ENABLE_PYTHON)
  find_package(SWIG)
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    set (ENABLE_PYTHON OFF)
    find_package(Python3 COMPONENTS Development)
	if (${SWIG_FOUND} AND ${Python3_FOUND})
	  message("Building interface for Python 3")
	  set (ENABLE_PYTHON ON)
	endif()
	find_package(Python2 COMPONENTS Development)
    if (${SWIG_FOUND} AND ${Python2_FOUND})
	  message("Building interface for Python 2")
	  set (ENABLE_PYTHON ON)
	endif()
  else()
    find_package(PythonLibs)
    if (${SWIG_FOUND} AND ${PYTHONLIBS_FOUND})
      message("Python support enabled")
    else()
      message("SWIG and/or python libraries missing. Python interface will not be built.")
      set(ENABLE_PYTHON OFF)
    endif()
  endif()
endif()

# enable all warnings
add_definitions( -Wall )
# add drop down menu for build type to gui
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "None" "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
endif(NOT CMAKE_BUILD_TYPE)

# add option to enable c++11 compiler features
option(ENABLE_CXX11 "Enable C++11" ON)
if(ENABLE_CXX11)
  set(CMAKE_CXX_STANDARD 11)
  message( "C++11 enabled")
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules)

# look for doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API Documentation with doxygen" VERBATIM
  )
else()
  message("Doxygen not found - doc building disabled")
endif()

include_directories( ${CMAKE_BINARY_DIR}/generated)

add_subdirectory(src)

option(ENABLE_EXAMPLES "Build examples" ON)
if(ENABLE_EXAMPLES)
add_subdirectory(examples)
endif()

install(
  EXPORT OPI-libs
  DESTINATION lib
)
