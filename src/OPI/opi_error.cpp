#include "opi_error.h"

namespace OPI
{
	const char* ErrorMessage(int code)
	{
		switch(code)
		{
			case NO_ERROR:
				return "No error";
			case CUDA_REQUIRED:
				return "CUDA Required";
			case CUDA_OLDVERSION:
				return "CUDA version too old";
			case INVALID_ARGUMENT:
				return "Invalid argument";
			case UNKNOWN_VARIABLE:
				return "Unknown variable";
			case INDEX_RANGE:
				return "Index out of range";
			case INVALID_TYPE:
				return "Invalid type";
			case NOT_IMPLEMENTED:
				return "Not implemented";
			case DIRECTORY_NOT_FOUND:
				return "Directory not found";
			case INVALID_DEVICE:
				return "Invalid device";
			case UNKNOWN_ERROR:
			default:
				return "Unknown error";
		}
	}

}

extern "C"
{
	const char* OPI_ErrorMessage(int code)
	{
		return OPI::ErrorMessage(code);
	}
}
