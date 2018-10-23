include_directories( ../src )
include_directories( ${CMAKE_BINARY_DIR}/src/OPI/)
include(examples)

# a simple cxx example
add_example(
  HostCXXExample
  SOURCES
    host_cpp_example.cpp
)

# a simple c example
add_example(
  HostCExample
  SOURCES
    host_c_example.c
)

# a simple fortran example
if(ENABLE_FORTRAN_SUPPORT)
  add_example(
    HostFortranExample
    FORTRAN
    SOURCES
      host_fortran_example.f90
  )
endif()

# a simple cpp plugin showing PropagatorProperties
add_example_plugin(
  PropagatorPropertiesCPP
  SOURCES
    propagator_properties_cpp.cpp
)

# a simple fortran example plugin
#if(ENABLE_FORTRAN_SUPPORT)
#  add_example_plugin(
#    PropagatorFortranExample
#    FORTRAN
#    SOURCES
#      propagator_fortran_example.f90
#  )
#endif()

# cpp basic example
add_example_plugin(
  PropagatorCPPBasic
  SOURCES
    propagator_basic_cpp.cpp
)

# cuda basic example
if(ENABLE_CUDA_SUPPORT)
  add_example_plugin(
    PropagatorCUDABasic
    CUDA
    SOURCES
      propagator_basic_cuda.cu
  )
endif()

# opencl basic example (C++ API)
if (ENABLE_CL_SUPPORT)
  add_example_plugin(
    PropagatorCL2Basic
    OPENCL
    SOURCES
      propagator_basic_cl2.cpp
  )
endif()

# cuda example
if(ENABLE_CUDA_SUPPORT)
  add_example_plugin(
    PropagatorCUDAMultiDevice
    CUDA
    SOURCES
      propagator_cuda_multidevice_example.cu
  )
endif()

# a simple OpenCL example plugin (C API)
if (ENABLE_CL_SUPPORT)
  add_example_plugin(
  	PropagatorCLExample
  	OPENCL
  	SOURCES
  		propagator_cl_example.cpp
  )
endif()

# a simple OpenCL example plugin (C++ API)
if (ENABLE_CL_SUPPORT)
  add_example_plugin(
    PropagatorCL2Example
    OPENCL
    SOURCES
     propagator_cl2_example.cpp
  )
endif()
