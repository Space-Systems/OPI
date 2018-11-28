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
#ifndef OPI_PLUGIN_H
#define OPI_PLUGIN_H
#include <string>
#include "../opi_error.h"

#include "../opi_plugininfo.h"
#include "opi_pluginprocs.h"

namespace OPI
{
	class DynLib;

	/// @cond INTERNAL_DOCUMENTATION
	/// @addtogroup OPI_INTERNAL
	/// @{
	/// The internal representation of a Plugin
	class Plugin
	{
		public:
			Plugin(DynLib* libhandle);
			~Plugin();

			/// Enable this plugin
			ErrorCode enable();
			/// Disable this plugin
			ErrorCode disable();

			/// Return the plugins name
            const char* getName() const;
			/// Return the plugins author
            const char* getAuthor() const;
			/// Return the plugins description
            const char* getDescription() const;
			/// Return the plugins information structure
			const PluginInfo& getInfo() const;
			/// Return the plugin shared object handle
			DynLib* getHandle() const;
		private:
			DynLib* handle;
			PluginInfo info;

			// function pointers
			pluginEnableFunction proc_enable;
			pluginDisableFunction proc_disable;
	};

	/// @}

	/// @endcond
}

#endif
