
include(GenerateType)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_CL_SUPPORT AND OPENCL_FOUND)
include_directories(${OpenCL_INCLUDE_DIR})
endif()

set( OPI_SOURCE_FILES
  opi_logger.cpp
  opi_population.cpp
  opi_perturbations.cpp
  opi_error.cpp
  opi_host.cpp
  opi_plugininfo.cpp
  opi_propagator.cpp
  opi_custom_propagator.cpp
  opi_query.cpp
  opi_indexpairlist.cpp
  opi_indexlist.cpp
  opi_collisiondetection.cpp
  opi_module.cpp

  opi_perturbation_module.cpp

  internal/opi_propagator_plugin.cpp
  internal/opi_query_plugin.cpp
  internal/opi_plugin.cpp
  internal/dynlib.cpp
  internal/miniz.c
  ${CMAKE_BINARY_DIR}/generated/OPI/opi_c_bindings.cpp
)

set( OPI_LIBRARY_HEADER
  # main api header files
  opi_cpp.h

  opi_common.h
  opi_error.h
  opi_datatypes.h
  opi_logger.h
  opi_population.h
  opi_perturbations.h
  opi_host.h
  opi_plugininfo.h
  opi_custom_propagator.h
  opi_implement_plugin.h
  opi_indexpairlist.h
  opi_indexlist.h
  opi_collisiondetection.h
  opi_module.h
  opi_gpusupport.h

  # plugin types
  opi_propagator.h
  opi_query.h
  opi_perturbation_module.h

  # helper templates
  opi_pimpl_helper.h

  # OpenCL functions
  opi_opencl_functions.h

  # auto-generated files for bindings
  ${CMAKE_BINARY_DIR}/generated/OPI/opi_types.h
  ${CMAKE_BINARY_DIR}/generated/OPI/opi_c_bindings.h
)

set( OPI_INTERNAL_HEADER
  # internal only headers - DO NOT INSTALL!
  internal/opi_propagator_plugin.h
  internal/opi_query_plugin.h
  internal/opi_pluginprocs.h
  internal/opi_plugin.h
  internal/opi_synchronized_data.h  
  internal/dynlib.h  
)

# we need the following for our compilation
add_definitions(
  -DOPI_COMPILING_DYNAMIC_LIBRARY
)

# create OPI library
add_library(
  OPI
  SHARED
  # source files
  ${OPI_SOURCE_FILES}

  # add header files so that they appear in project views of
  # various IDEs
  ${OPI_LIBRARY_HEADER}
  ${OPI_INTERNAL_HEADER}


  # header files
)

set( OPI_BINDINGS
  # types MUST BE the first file to parse
  bindings/types.cmake
  # now add all class bindings
  bindings/host.cmake
  bindings/objectdata.cmake
  bindings/propagator.cmake
  bindings/module.cmake
)
# cpp needs the generated type information
generate_bindings(CPP     bindings/types.cmake)

generate_bindings(C       ${OPI_BINDINGS})
if(ENABLE_FORTRAN_SUPPORT)
  generate_bindings(FORTRAN ${OPI_BINDINGS})
  add_library(
    OPI-Fortran
    SHARED
    ${CMAKE_BINARY_DIR}/generated/OPI/opi_fortran_bindings.f90
  )
endif()

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
  string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
  set_target_properties( OPI PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/ )
  set_target_properties( OPI PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/${OUTPUTCONFIG}/ )
endforeach( )

if(NOT WIN32)
  # link with libraries
  target_link_libraries( OPI
    dl
  )
if(ENABLE_FORTRAN_SUPPORT)
  target_link_libraries( OPI-Fortran
    OPI
    dl
  )
  SET_TARGET_PROPERTIES (OPI OPI-Fortran
    PROPERTIES BUILD_WITH_INSTALL_RPATH 1
               INSTALL_NAME_DIR "@loader_path/../lib"
  )
else()
  SET_TARGET_PROPERTIES (OPI
    PROPERTIES BUILD_WITH_INSTALL_RPATH 1
               INSTALL_NAME_DIR "@loader_path/../lib"
  )
endif()
endif()

# install library
install(
  TARGETS OPI
  EXPORT OPI-libs
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
)

# install library header files
install(
  FILES
  ${OPI_LIBRARY_HEADER}
  DESTINATION include/OPI/
)

if(ENABLE_FORTRAN_SUPPORT)
  install(
    FILES
      ${CMAKE_BINARY_DIR}/fortran_modules/opi.mod
      ${CMAKE_BINARY_DIR}/fortran_modules/opi_types.mod
    DESTINATION include/OPI/fortran/
  )
  install(
    TARGETS OPI-Fortran
    EXPORT OPI-libs
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
  )
endif()


if (ENABLE_PYTHON)
  # Path to SWIG interface file
  set (OPI_INTERFACE_FILE bindings/opi.i)
  # Include SWIG-specific CMake commands.
  if (${SWIG_USE_FILE})
    include(${SWIG_USE_FILE})
  else()
    include(UseSWIG)
  endif()
  # Put SWIG in C++ mode.
  set_source_files_properties(${OPI_INTERFACE_FILE} PROPERTIES CPLUSPLUS ON)
  include_directories(${CMAKE_BINARY_DIR}/generated/OPI)

  if (${CMAKE_VERSION} VERSION_LESS "3.12")
    # Add Python headers and generated OPI headers to the include path.
    include_directories(${PYTHON_INCLUDE_PATH})
    # Create python module from interface definition file.
    set(OPI_PY_MODULE_NAME OPI_py${PYTHONLIBS_VERSION_STRING})
    if (${CMAKE_VERSION} VERSION_LESS "3.8")
      swig_add_module(${OPI_PY_MODULE_NAME} python ${OPI_INTERFACE_FILE})
    else()
      swig_add_library(${OPI_PY_MODULE_NAME}
	  TYPE MODULE
	  LANGUAGE python
	  SOURCES ${OPI_INTERFACE_FILE}
	  )
    endif()
    # Link Python module to the library we created earlier,
    # as well as the necessary Python libraries.
    swig_link_libraries(${OPI_PY_MODULE_NAME} ${PYTHON_LIBRARIES} OPI)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OPI.py DESTINATION lib/python${PYTHONLIBS_VERSION_STRING})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI_py${PYTHONLIBS_VERSION_STRING}.pyd DESTINATION lib/python${PYTHONLIBS_VERSION_STRING}/ RENAME _OPI.pyd OPTIONAL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI_py${PYTHONLIBS_VERSION_STRING}.so DESTINATION lib/python${PYTHONLIBS_VERSION_STRING}/ RENAME _OPI.so OPTIONAL)
  else()
    if (Python3_FOUND)
      set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/python3)
      set_source_files_properties(${OPI_INTERFACE_FILE} PROPERTIES GENERATED_INCLUDE_DIRECTORIES ${Python3_INCLUDE_DIRS})
      set_source_files_properties(${OPI_INTERFACE_FILE} PROPERTIES DEPENDS ${CMAKE_BINARY_DIR}/generated/OPI/opi_types.h)
      swig_add_library(OPI
          TYPE MODULE
          LANGUAGE python
          SOURCES ${OPI_INTERFACE_FILE})
      swig_link_libraries(OPI ${Python3_LIBRARIES} OPI)

      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python3/OPI.py DESTINATION lib/python3)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI.pyd DESTINATION lib/python3/ OPTIONAL)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI.so DESTINATION lib/python3/ OPTIONAL)
    elseif (Python2_FOUND)
      set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/python2)
      set_source_files_properties(${OPI_INTERFACE_FILE} PROPERTIES GENERATED_INCLUDE_DIRECTORIES ${Python2_INCLUDE_DIRS})
      set_source_files_properties(${OPI_INTERFACE_FILE} PROPERTIES DEPENDS ${CMAKE_BINARY_DIR}/generated/OPI/opi_types.h)
      swig_add_library(OPI
          TYPE MODULE
          LANGUAGE python
          SOURCES ${OPI_INTERFACE_FILE})
      swig_link_libraries(OPI ${Python2_LIBRARIES} OPI)

      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python2/OPI.py DESTINATION lib/python2)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI.pyd DESTINATION lib/python2/ OPTIONAL)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_OPI.so DESTINATION lib/python2/ OPTIONAL)
    endif()
  endif()
endif()
