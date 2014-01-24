#ifndef OPI_ERROR_H
#define OPI_ERROR_H
#include "opi_common.h"
#include "opi_datatypes.h"
#ifdef __cplusplus
extern "C" {
#endif

OPI_API_EXPORT const char* OPI_ErrorMessage(int code);
#ifdef __cplusplus
}
namespace OPI
{
	OPI_API_EXPORT const char* ErrorMessage(int code);
}
#endif


#endif
