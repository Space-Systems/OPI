include(ParseArguments)
macro(add_example APPLICATION)
  PARSE_ARGUMENTS( ARG
    "SOURCES;LIBRARIES"
    "C;FORTRAN"
    ${ARGN} )
option(EXAMPLES_${APPLICATION} "Build example \"${APPLICATION}\"" ON)
if(EXAMPLES_${APPLICATION})
  set(TARGET_DISABLED FALSE)
  set(TARGET_LANG CXX)
  set(ADD_LIBS)
  if(ARG_FORTRAN)
    if(NOT ENABLE_FORTRAN_SUPPORT)
      set(TARGET_DISABLED TRUE)
    endif()
    set(TARGET_LANG Fortran)
    set(ADD_LIBS "OPI-Fortran")
  endif()
  if(ARG_C)
    set(TARGET_LANG C)
  endif()
  if(ARG_CUDA)
    if(NOT ENABLE_CUDA_SUPPORT)
      set(TARGET_DISABLED TRUE)
    endif()
    set(TARGET_LANG CUDA)
  endif()
  if(NOT TARGET_DISABLED)
    add_executable(
      ${APPLICATION}
      ${ARG_SOURCES}
    )

    target_link_libraries(
    ${APPLICATION}
      OPI
      ${ADD_LIBS}
    )

    set_target_properties( ${APPLICATION} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )
  foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
      set_target_properties( ${APPLICATION} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/ )
      set_target_properties( ${APPLICATION} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/ )
    endforeach( )
  endif()
endif()
endmacro()

macro(add_example_plugin PLUGIN)
  PARSE_ARGUMENTS( ARG
    "SOURCES;LIBRARIES"
    "CUDA;OPENCL;FORTRAN"
    ${ARGN} )
option(EXAMPLES_${PLUGIN} "Build example plugin \"${PLUGIN}\"" ON)
if(EXAMPLES_${PLUGIN})
  set(TARGET_DISABLED FALSE)
  if(ARG_FORTRAN)
    if(NOT ENABLE_FORTRAN_SUPPORT)
      set(TARGET_DISABLED TRUE)
    endif()
    set(ARG_LIBRARIES ${ARG_LIBRARIES} OPI-Fortran)
  endif()
  if(ARG_CUDA)
    if(NOT ENABLE_CUDA_SUPPORT)
      set(TARGET_DISABLED TRUE)
    endif()
  endif()
  if (ARG_OPENCL)
	if (ENABLE_CL_SUPPORT)
		include_directories(${OpenCL_INCLUDE_DIR})
		set(ARG_LIBRARIES ${ARG_LIBRARIES} ${OpenCL_LIBRARY} OPI-cl)
	endif()
  endif()
  if( NOT TARGET_DISABLED)
    add_library(
      ${PLUGIN}
      MODULE
      ${ARG_SOURCES}
    )

    if(TARGET ${PLUGIN})
      target_link_libraries(
        ${PLUGIN}
        OPI
        ${ARG_LIBRARIES}
      )

      set_target_properties( ${PLUGIN} PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/plugins
      )
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
      string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        set_target_properties( ${PLUGIN} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/ )
        set_target_properties( ${PLUGIN} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/plugins/ )
      endforeach( )
  endif()
  endif()
endif()
endmacro()
