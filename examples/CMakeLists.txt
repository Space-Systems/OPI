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
add_example(
  HostFortranExample
  FORTRAN
  SOURCES
    host_fortran_example.f90
)

# a simple cpp example plugins
add_example_plugin(
  PropagatorCPPExample
  SOURCES
    propagator_cpp_example.cpp
)

# a simple fortran example plugin
add_example_plugin(
  PropagatorFortranExample
  FORTRAN
  SOURCES
    propagator_fortran_example.f90
)

# cpp basic example
add_example_plugin(
  PropagatorCPPBasic
  SOURCES
    propagator_basic_cpp.cpp
)

# cuda basic example
add_example_plugin(
  PropagatorCUDABasic
  CUDA
  SOURCES
    propagator_basic_cuda.cu
)

# opencl basic example (C++ API)
add_example_plugin(
  PropagatorCL2Basic
  OPENCL
  SOURCES
    propagator_basic_cl2.cpp
)

# cuda example
add_example_plugin(
  PropagatorCUDAMultiDevice
  CUDA
  SOURCES
    propagator_cuda_multidevice_example.cu
)

# a simple OpenCL example plugin (C API)
add_example_plugin(
	PropagatorCLExample
	OPENCL
	SOURCES
		propagator_cl_example.cpp
)

# a simple OpenCL example plugin (C++ API)
add_example_plugin(
  PropagatorCL2Example
  OPENCL
  SOURCES
    propagator_cl2_example.cpp
)
