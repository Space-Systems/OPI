#include "opi_error.h"

namespace OPI
{
	const char* ErrorMessage(int code)
	{
		switch(code)
		{
			case SUCCESS:
				return "No error";
			case INVALID_ARGUMENT:
				return "Invalid argument";
			case UNKNOWN_VARIABLE:
				return "Unknown variable";
			case INDEX_RANGE:
				return "Index out of range";
			case INVALID_TYPE:
				return "Invalid type";
			case DIRECTORY_NOT_FOUND:
				return "Directory not found";
			case INVALID_DEVICE:
				return "Invalid device";
			case INVALID_PROPERTY:
                return "Invalid property / missing config file";
			case INCOMPATIBLE_TYPES:
				return "INCOMPATIBLE_TYPES";
			case NOT_IMPLEMENTED:
				return "Not implemented";
			case CUDA_REQUIRED:
				return "CUDA Required";
			case CUDA_OLDVERSION:
				return "CUDA version too old";
			case UNKNOWN_ERROR:
				break;
//			default:

		}
		return "Unknown error";
	}
}

extern "C"
{
	const char* OPI_ErrorMessage(int code)
	{
		return OPI::ErrorMessage(code);
	}
}
