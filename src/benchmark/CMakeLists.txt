
if(UNIX)
  add_executable( benchmark
    benchmark.cpp
  )

  target_link_libraries( benchmark OPI rt)
endif()
