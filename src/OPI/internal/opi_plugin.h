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
            std::string getName() const;
			/// Return the plugins author
            std::string getAuthor() const;
			/// Return the plugins description
            std::string getDescription() const;
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
