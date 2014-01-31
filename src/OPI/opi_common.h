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
#ifndef OPI_COMMON_COMMON_H
#define OPI_COMMON_COMMON_H

const int OPI_API_VERSION_MAJOR = 0;
const int OPI_API_VERSION_MINOR = 1;
#if WIN32
#ifdef OPI_COMPILING_DYNAMIC_LIBRARY
#define OPI_API_EXPORT __declspec( dllexport )
#else
#define OPI_API_EXPORT __declspec( dllimport )
#endif
#else
#define OPI_API_EXPORT
#endif

#endif
