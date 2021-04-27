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
            virtual ErrorCode runPropagation(Population& population, JulianDay epoch, long long dt, PropagationMode mode, IndexList* indices);
			virtual int requiresCUDA();
		private:
			Plugin* plugin;
			// propagate proc
			pluginPropagateFunction proc_propagate;
			pluginInitFunction proc_init;

	};

	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}


#endif
