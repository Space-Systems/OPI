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
    typedef ErrorCode (*pluginPropagateFunction)(OPI_Propagator propagator, OPI_Population population, JulianDay epoch, long dt, PropagationMode mode, IndexList* indices);

	// cpp distance query interface function
	typedef DistanceQuery* (*pluginDistanceQueryFunction)(OPI_Host host);

	// cpp collision detection interface function
	typedef CollisionDetection* (*pluginCollisionDetectionFunction)(OPI_Host host);
	}
}

#endif
