
# main library
add_subdirectory(OPI)

include_directories(
  # change this if you want to test the install script
  #${CMAKE_INSTALL_PREFIX}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# cuda support plugin
if(ENABLE_CUDA_SUPPORT)
  add_subdirectory(cuda_support)
endif()

export(
  TARGETS
    OPI

  FILE ${CMAKE_BINARY_DIR}/OPI-libs.cmake
)

option(ENABLE_BENCHMARK "Build benchmarking program" OFF)
if(ENABLE_BENCHMARK)
  add_subdirectory(benchmark)
endif()
