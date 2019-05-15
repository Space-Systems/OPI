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
#include "opi_plugin.h"
#include "dynlib.h"
#include <iostream>
#include <cstring>
namespace OPI
{
	/// @cond INTERNAL_DOCUMENTATION
	Plugin::Plugin(DynLib *libhandle)
	{
		handle = libhandle;
		// load information function from library
		pluginInfoFunction func = (pluginInfoFunction)handle->loadFunction("OPI_Plugin_info");
		info.name = "Unset plugin name";
		info.name_len = strlen(info.name);
		info.author = "Unset plugin author";
		info.author_len = strlen(info.author);
		info.desc = "Unset plugin desc";
		info.desc_len = strlen(info.desc);
		info.type = OPI_UNKNOWN_PLUGIN;
		if(func)
		{
			func(&info);
		// add verification of api version here

			// load
			proc_enable = (pluginEnableFunction)handle->loadFunction("OPI_Plugin_enable", true);
			proc_disable = (pluginDisableFunction)handle->loadFunction("OPI_Plugin_disable", true);
		}
	}

	Plugin::~Plugin()
	{
		delete handle;
	}

	DynLib* Plugin::getHandle() const
	{
		return handle;
	}

	const PluginInfo& Plugin::getInfo() const
	{
		return info;
	}

	ErrorCode Plugin::enable()
	{
		if(proc_enable)
			return proc_enable(this);
		return SUCCESS;
	}

	ErrorCode Plugin::disable()
	{
		if(proc_disable)
			return proc_disable(this);
		return SUCCESS;
	}

    const char* Plugin::getName() const
	{
        return std::string(info.name, info.name_len).c_str();
	}

    const char* Plugin::getAuthor() const
	{
        return std::string(info.author, info.author_len).c_str();
	}

    const char* Plugin::getDescription() const
	{
        return std::string(info.desc, info.desc_len).c_str();
	}
	/// @endcond
}
