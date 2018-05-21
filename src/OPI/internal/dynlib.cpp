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
#include "dynlib.h"
#include <iostream>
namespace OPI
{

	DynLib::DynLib(const std::string& name, bool silentfail )
	{
		handle = 0;
#ifdef _WIN32
		std::string buffer = name;
		handle = LoadLibrary(buffer.c_str());
		if((!handle) && (!silentfail))
		{
			std::cout << "Error loading library " << name << ": " << GetLastError() << std::endl;
		}
#else
		std::string buffer = name;
		handle = dlopen(buffer.c_str(), RTLD_LAZY);
		if((!handle) && (!silentfail))
		{
			std::cout << dlerror() << std::endl;
		}
#endif
	}

	void* DynLib::loadFunction(const char *name, bool silentfail)
	{
		if(handle == 0)
			return 0;
#ifdef _WIN32
		void* result = 0;
		result = (void*)GetProcAddress(handle, name);
		if(!result)
		{
			if(!silentfail)
				std::cout << "Error loading Function " << name << std::endl;
		}
#else
		void* result = 0;
		char* error;
		result = dlsym(handle, name);
		if((error = dlerror()) != NULL)
		{
			if(!silentfail)
				std::cout << "Error loading Function " << name << ": " << error << std::endl;
		}
#endif
		return result;
	}

	bool DynLib::isValid() const
	{
		return (handle != 0);
	}

	std::string DynLib::getSuffix()
	{
#ifdef _WIN32
		return ".dll";
#else
		return ".so";
#endif
	}

	DynLib::~DynLib()
	{
		if(handle != 0)
		{
#ifdef _WIN32
			FreeLibrary(handle);
#else
			dlclose(handle);
#endif
		}
	}
}
