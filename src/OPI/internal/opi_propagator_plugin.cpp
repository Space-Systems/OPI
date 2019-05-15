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
#include "opi_propagator_plugin.h"
#include "opi_plugin.h"
#include "dynlib.h"
namespace OPI
{
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	PropagatorPlugin::PropagatorPlugin(Plugin *_plugin)
	{
		plugin = _plugin;

		DynLib* handle = plugin->getHandle();
		proc_init = (pluginInitFunction)(handle->loadFunction("OPI_Plugin_init", true));
		if(proc_init)
		{
			proc_init(this);
		}
		proc_propagate = (pluginPropagateFunction)(handle->loadFunction("OPI_Plugin_propagate", true));
		setName(plugin->getName());
		setAuthor(plugin->getAuthor());
		setDescription(plugin->getDescription());
	}

	PropagatorPlugin::~PropagatorPlugin()
	{

	}

	ErrorCode PropagatorPlugin::enable()
	{
		return plugin->enable();
	}

	ErrorCode PropagatorPlugin::disable()
	{
		return plugin->disable();
	}

    ErrorCode PropagatorPlugin::runPropagation(Population& population, double julian_day, double dt, PropagationMode mode, IndexList* indices)
	{
		if(proc_propagate)
            return proc_propagate(this, (void*)(&population), julian_day, dt, mode, indices);
		return NOT_IMPLEMENTED;
	}

	int PropagatorPlugin::requiresCUDA()
	{
		return 0;
	}

    //FIXME Implement missing interface functions

	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}
