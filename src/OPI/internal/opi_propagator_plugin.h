#ifndef OPI_HOST_PROPAGATOR_PLUGIN_H
#define OPI_HOST_PROPAGATOR_PLUGIN_H
#include "../opi_propagator.h"
#include "opi_pluginprocs.h"
namespace OPI
{
	class Plugin;

	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	class PropagatorPlugin:
			public Propagator
	{
		public:
			PropagatorPlugin(Plugin* _plugin);
			~PropagatorPlugin();


			virtual ErrorCode enable();
			virtual ErrorCode disable();
			virtual ErrorCode runPropagation(ObjectData& data, float years, float seconds, float dt );
			virtual ErrorCode runIndexedPropagation(ObjectData& data, int* indices, int index_size, float years, float seconds, float dt );
		private:
			Plugin* plugin;
			// propagate proc
			pluginPropagateFunction proc_propagate;
			pluginPropagateFunctionIndexed proc_propagate_indexed;

	};

	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}


#endif
