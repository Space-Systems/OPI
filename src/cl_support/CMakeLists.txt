# include cuda sdk directories
include_directories(${OpenCL_INCLUDE_DIR})

add_library(
  OPI-cl
  SHARED
  opi_cl_support.cpp
)

set_target_properties( OPI-cl PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/plugins/support
)

target_link_libraries( OPI-cl ${OpenCL_LIBRARY})

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
  string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
  set_target_properties( OPI-cl PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/support )
  set_target_properties( OPI-cl PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/support )
endforeach( )

install(
  TARGETS OPI-cl
  DESTINATION plugins/support
)
