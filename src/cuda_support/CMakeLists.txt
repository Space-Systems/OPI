# include cuda sdk directories
include_directories(${CUDA_INCLUDE_DIRS})

add_library(
  OPI-cuda
  MODULE
  opi_cuda_support.cpp ../OPI/opi_logger.cpp
)

set_target_properties( OPI-cuda PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/plugins/support
)

target_link_libraries( OPI-cuda ${CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
  string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
  set_target_properties( OPI-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/support )
  set_target_properties( OPI-cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/support )
endforeach( )

install(
  TARGETS OPI-cuda
  LIBRARY DESTINATION plugins/support
)
