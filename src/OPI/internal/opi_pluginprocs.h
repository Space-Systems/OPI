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
#ifndef OPI_HOST_PLUGIN_PROCS_H
#define OPI_HOST_PLUGIN_PROCS_H
#include "../opi_population.h"
#include "../opi_error.h"
#include "../opi_plugininfo.h"

typedef void* OPI_Module;
typedef void* OPI_Propagator;
typedef void* OPI_DistanceQuery;
typedef void* OPI_CollisionDetection;
namespace OPI
{
	class Propagator;
	class DistanceQuery;
	class CollisionDetection;

	extern "C"
	{
	typedef void* OPI_Population;
	typedef void (*pluginInfoFunction)(PluginInfo* info);
	//! plugin disable function
	typedef ErrorCode (*pluginEnableFunction)(OPI_Module module);
	//! plugin enable function
	typedef ErrorCode (*pluginDisableFunction)(OPI_Module module);

	// generic interface functions
	typedef void (*pluginInitFunction)(OPI_Module module);

	// cpp propagator interface function
	typedef Propagator* (*pluginPropagatorFunction)(OPI_Host host);
	// c interface propagation function
    typedef ErrorCode (*pluginPropagateFunction)(OPI_Propagator propagator, OPI_Population data, double julian_day, double dt, PropagationMode mode, IndexList* indices);

	// cpp distance query interface function
	typedef DistanceQuery* (*pluginDistanceQueryFunction)(OPI_Host host);

	// cpp collision detection interface function
	typedef CollisionDetection* (*pluginCollisionDetectionFunction)(OPI_Host host);
	}
}

#endif
