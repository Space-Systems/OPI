#ifndef OPI_HOST_PLUGIN_PROCS_H
#define OPI_HOST_PLUGIN_PROCS_H
#include "../opi_data.h"
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
	typedef void* OPI_ObjectData;
	typedef void (*pluginInfoFunction)(PluginInfo* info);
	//! plugin disable function
	typedef ErrorCode (*pluginEnableFunction)(OPI_Module module);
	//! plugin enable function
	typedef ErrorCode (*pluginDisableFunction)(OPI_Module module);

	// cpp propagator interface function
	typedef Propagator* (*pluginPropagatorFunction)(OPI_Host host);
	// c interface propagation function
	typedef ErrorCode (*pluginPropagateFunction)(OPI_Propagator propagator, OPI_ObjectData data, float years, float seconds, float dt);

	// c interface propagation function
	typedef ErrorCode (*pluginPropagateFunctionIndexed)(OPI_Propagator propagator, OPI_ObjectData data, int* indices, int index_size, float years, float seconds, float dt);

	// cpp distance query interface function
	typedef DistanceQuery* (*pluginDistanceQueryFunction)(OPI_Host host);

	// cpp collision detection interface function
	typedef CollisionDetection* (*pluginCollisionDetectionFunction)(OPI_Host host);
	}
}

#endif
