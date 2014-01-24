macro(generate_bindings TYPE_LANG )
  set(RUN_SCRIPT)
  set(OUTPUT_FILES)
  set(OUTPUT_PATH ${CMAKE_BINARY_DIR}/generated/OPI/)
  if(${TYPE_LANG} STREQUAL "C")
    set(RUN_SCRIPT generate_c)
    set(OUTPUT_FILES ${OUTPUT_PATH}/opi_c_bindings.h ${OUTPUT_PATH}/opi_c_bindings.cpp)
  elseif(${TYPE_LANG} STREQUAL "CPP")
    set(RUN_SCRIPT generate_cxx)
    set(OUTPUT_FILES ${OUTPUT_PATH}/opi_types.h)
  elseif(${TYPE_LANG} STREQUAL "FORTRAN")
    set(RUN_SCRIPT generate_fortran)
    set(OUTPUT_FILES ${OUTPUT_PATH}/opi_fortran_bindings.f90)
  endif()
  if(RUN_SCRIPT)
    add_custom_command(
      OUTPUT ${OUTPUT_FILES}
      COMMAND ${CMAKE_COMMAND} -DPROCESS_FILES="${ARGN}" -DOUTPUT_PATH=${OUTPUT_PATH} -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH} -P ${CMAKE_SOURCE_DIR}/cmake/modules/${RUN_SCRIPT}.cmake
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating ${TYPE_LANG} bindings"
      MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/cmake/modules/${RUN_SCRIPT}.cmake
      DEPENDS ${ARGN}
    )
  else()
    message(FATAL_ERROR "Unknown language: ${TYPE_LANG}")
  endif()
endmacro()

# just a dummy target to add generate_typedef.cmake to the projects file list
add_custom_target(DUMMY
  ALL
  SOURCES
    ${CMAKE_SOURCE_DIR}/cmake/modules/generate_c.cmake
    ${CMAKE_SOURCE_DIR}/cmake/modules/generate_cxx.cmake
    ${CMAKE_SOURCE_DIR}/cmake/modules/generate_fortran.cmake
)
