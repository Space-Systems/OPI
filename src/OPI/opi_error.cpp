/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
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
