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
#pragma once
#ifndef OPI_DYNAMIC_LIBRARIES_H
#define OPI_DYNAMIC_LIBRARIES_H
#ifdef WIN32
#include <windows.h>
typedef HINSTANCE DynLibHandle;
#else
#include <dlfcn.h>
typedef void* DynLibHandle;
#endif
#include <string>
namespace OPI
{
	/**
	 * @brief A platform-independent wrapper for shared object loading
	 * @ingroup CPP_API_INTERNAL_GROUP
	 */
	class DynLib
	{
		public:
			/**
			 * @brief Loads a shared object by its filename
			 * @param name Filename
			 * @param silentfail If false gives some information to stdout
			 */
			DynLib(const std::string& name, bool silentfail = false);
			~DynLib();
			/**
			 * @brief Loads a function from the shared object
			 * @param name Function name to load
			 * @param silentfail If false gives some information to stdout
			 * @return
			 */
			void* loadFunction(const char* name, bool silentfail = false);
			/**
			 * @brief Check if the shared object was loaded successfully
			 * @return Success
			 */
			bool isValid() const;
			/**
			 * @brief Returns the platforms shared object suffix
			 * @return Suffix (.dll/.so/...)
			 */
			static std::string getSuffix();
		private:
			DynLibHandle handle;
	};
}
#endif
