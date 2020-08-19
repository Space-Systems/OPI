#ifndef OPI_COMMON_COMMON_H
#define OPI_COMMON_COMMON_H

const int OPI_API_VERSION_MAJOR = 2019;
const int OPI_API_VERSION_MINOR = 8;
#ifdef WIN32
#ifdef OPI_COMPILING_DYNAMIC_LIBRARY
#define OPI_API_EXPORT __declspec( dllexport )
#else
#define OPI_API_EXPORT __declspec( dllimport )
#endif
#else
#define OPI_API_EXPORT
#endif

#endif
