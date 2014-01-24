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
		proc_propagate = (pluginPropagateFunction)(handle->loadFunction("OPI_Plugin_propagate", true));
		proc_propagate_indexed = (pluginPropagateFunctionIndexed)(handle->loadFunction("OPI_Plugin_propagateIndexed", true));
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

	ErrorCode PropagatorPlugin::runPropagation(ObjectData& data, float years, float seconds, float dt)
	{
		if(proc_propagate)
			return proc_propagate(this, (void*)(&data), years, seconds, dt);
		return NOT_IMPLEMENTED;
	}

	ErrorCode PropagatorPlugin::runIndexedPropagation(ObjectData& data, int *indices, int index_size, float years, float seconds, float dt)
	{
		if(proc_propagate_indexed)
			return proc_propagate_indexed(this, &data, indices, index_size, years, seconds, dt);
		return NOT_IMPLEMENTED;
	}

	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}
