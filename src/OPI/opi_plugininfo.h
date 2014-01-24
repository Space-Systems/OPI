#ifndef OPI_COMMON_PLUGININFO_H
#define OPI_COMMON_PLUGININFO_H
#ifdef __cplusplus
extern "C" {
#endif

enum OPI_PluginType
{
	OPI_UNKNOWN_PLUGIN = 0,
	OPI_PROPAGATOR_PLUGIN = 1,
	OPI_PROPAGATOR_MODULE_PLUGIN = 2,
	OPI_PROPAGATOR_INTEGRATOR_PLUGIN = 3,

	OPI_DISTANCE_QUERY_PLUGIN = 10,
	OPI_COLLISION_DETECTION_PLUGIN = 20,
	OPI_COLLISION_HANDLING_PLUGIN = 30
};

/**
 * @brief The OPI_PluginInfo struct contains general information about a plugin
 */
struct OPI_PluginInfo
{
		// used api version
		int apiVersionMajor;
		int apiVersionMinor;
		// plugin name
		const char* name;
		// plugin name length
		int name_len;
		// plugin author
		const char* author;
		// plugin author length
		int author_len;
		// plugin description
		const char* desc;
		// plugin desc length
		int desc_len;
		// plugin version
		int versionMajor;
		int versionMinor;
		int versionPatch;
		int type;
		bool cppPlugin;
};

#ifdef __cplusplus
}

namespace OPI
{
	typedef OPI_PluginType PluginType;
	typedef OPI_PluginInfo PluginInfo;
}
#endif

#endif
