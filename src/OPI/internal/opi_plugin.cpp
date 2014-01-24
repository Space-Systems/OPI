#include "opi_plugin.h"
#include "dynlib.h"
#include <iostream>
namespace OPI
{
	/// @cond INTERNAL_DOCUMENTATION
	Plugin::Plugin(DynLib *libhandle)
	{
		handle = libhandle;
		// load information function from library
		pluginInfoFunction func = (pluginInfoFunction)handle->loadFunction("OPI_Plugin_info");
		info.name = "Unset plugin name";
		info.author = "Unset plugin author";
		info.desc = "Unset plugin desc";
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
		return NO_ERROR;
	}

	ErrorCode Plugin::disable()
	{
		if(proc_disable)
			return proc_disable(this);
		return NO_ERROR;
	}

	std::string Plugin::getName() const
	{
		return std::string(info.name, info.name_len);
	}

	std::string Plugin::getAuthor() const
	{
		return std::string(info.author, info.author_len);
	}

	std::string Plugin::getDescription() const
	{
		return std::string(info.desc, info.desc_len);
	}
	/// @endcond
}
