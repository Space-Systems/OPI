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
        setName(plugin->getName().c_str());
        setAuthor(plugin->getAuthor().c_str());
        setDescription(plugin->getDescription().c_str());
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

    ErrorCode PropagatorPlugin::runPropagation(Population& population, JulianDay epoch, long long dt, PropagationMode mode, IndexList* indices)
	{
		if(proc_propagate)
            return proc_propagate(this, (void*)(&population), epoch, dt, mode, indices);
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
