#include "opi_host.h"
#include "internal/opi_plugin.h"
#include "opi_logger.h"
#include "opi_propagator.h"
#include "opi_gpusupport.h"
#include "internal/opi_propagator_plugin.h"
#include "internal/opi_query_plugin.h"
#include "opi_custom_propagator.h"
#include "opi_collisiondetection.h"
#include "internal/dynlib.h"
#include <iostream>
#include <cstring>
#ifdef _MSC_VER
#include "internal/msdirent.h"
#else
// mingw should have dirent
#include <dirent.h>
#endif
namespace OPI
{
	//! \cond INTERNAL_DOCUMENTATION

	class HostImpl
	{
		public:
			GpuSupport* gpuSupport;
			DynLib* gpuSupportPluginHandle;

			std::vector<Plugin*> pluginlist;
			std::vector<Propagator*> propagagorlist;
			std::vector<PerturbationModule*> modulelist;
			std::vector<PropagatorIntegrator*> integratorlist;
			std::vector<DistanceQuery*> querylist;
			std::vector<CollisionDetection*> detectionlist;
            std::string pluginPath;
			OPI_ErrorCallback errorCallback;
			void* errorCallbackParameter;
			mutable ErrorCode lastError;
	};

	//! \endcond

    Host::Host()
	{
        impl->errorCallback = 0;
		impl->lastError = SUCCESS;
		impl->errorCallbackParameter = 0;

		impl->gpuSupport = 0;
		impl->gpuSupportPluginHandle = 0;

        impl->pluginPath = "";
	}

	Host::~Host()
	{
		// free every propagator
		for(size_t i = 0; i < impl->propagagorlist.size(); ++i)
		{
			impl->propagagorlist[i]->disable();
			delete impl->propagagorlist[i];
		}

		for(size_t i = 0; i < impl->querylist.size(); ++i)
		{
			impl->querylist[i]->disable();
			delete impl->querylist[i];
		}

		for(size_t i = 0; i < impl->detectionlist.size(); ++i)
		{
			impl->detectionlist[i]->disable();
			delete impl->detectionlist[i];
		}

		// now free the support plugin memory
		if(impl->gpuSupport)
			impl->gpuSupport->shutdown();
		delete impl->gpuSupport;
		delete impl->gpuSupportPluginHandle;

		// free all plugin handles
		for(size_t i = 0; i < impl->pluginlist.size(); ++i)
			delete impl->pluginlist[i];
	}

    void Host::logToFile(const char* logfileName, bool append)
    {
		if (logfileName == "")
			Logger::setMode(Logger::LOGMODE_STDOUT);
		else
			Logger::setMode(Logger::LOGMODE_FILE, logfileName, nullptr, append);
    }

	void Host::setVerboseLevel(int level)
	{
		Logger::setVerboseLevel(level);
	}

	void Host::setErrorCallback(OPI_ErrorCallback callback, void* privatedata)
	{
		impl->errorCallback = callback;
		impl->errorCallbackParameter = privatedata;
	}

	ErrorCode Host::getLastError() const
	{
		return impl->lastError;
	}

    ErrorCode Host::loadPlugins(const char* plugindir, gpuPlatform platformSupport, int platformNumber, int deviceNumber)
	{
		ErrorCode status = SUCCESS;
		Logger::out(0) << "OPI " << OPI_API_VERSION_MAJOR << "." << OPI_API_VERSION_MINOR << std::endl;
		Logger::out(0) << "Loading plugins from " << plugindir << std::endl;
        std::string suffix = DynLib::getSuffix();

		// check if the cuda support plugin is loaded
		if(impl->gpuSupport == 0 && platformSupport != PLATFORM_NONE)
		{
			// try to load cuda plugin
			std::string pluginName = (platformSupport == PLATFORM_OPENCL ? "OPI-cl" : "OPI-cuda");
			std::string gpuFrameworkName = (platformSupport == PLATFORM_OPENCL ? "OpenCL" : "CUDA");
            std::string libraryFileName = std::string(plugindir) + "/support/" + pluginName + suffix;
			Logger::out(0) << "Loading support library " << libraryFileName << std::endl;

			impl->gpuSupportPluginHandle = new DynLib(libraryFileName, true);
			if(impl->gpuSupportPluginHandle)
			{
				procCreateGpuSupport proc_create_support = (procCreateGpuSupport)impl->gpuSupportPluginHandle->loadFunction("createGpuSupport");
				if(proc_create_support)
				{
					// plugin successfully loaded
					impl->gpuSupport = proc_create_support();
                    impl->gpuSupport->init(platformNumber, deviceNumber);
				}
				else {
					platformSupport = PLATFORM_NONE;
#ifdef WIN32
					DWORD errorMessageID = ::GetLastError();
					LPSTR messageBuffer = nullptr;
					size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
						NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

					std::string message(messageBuffer, size);
#else
					std::string message = "";
#endif
					Logger::out(0) << "[OPI] Unable to load GPU support library. " << message << std::endl;
				}
			}
			else {
				platformSupport = PLATFORM_NONE;
				Logger::out(0) << "[OPI] Cannot find GPU support library (" << libraryFileName << ")."<< std::endl
					<< "Check your path and make sure your " << gpuFrameworkName << " drivers are installed correctly." << std::endl;
			}
		}

		// now load all plugins from plugindir
        DIR* dir = opendir(plugindir);
		// check if plugindir is a directory
		if(dir != 0)
		{
            impl->pluginPath = std::string(plugindir);
			dirent* dir_entry = 0;
			// iterate over all directory entries
			while((dir_entry = readdir(dir)))
			{
				std::string entry_name(dir_entry->d_name);
                // only consider files ending in the platform-specific library suffix
                if(entry_name.find_last_of(".") != std::string::npos &&
                   entry_name.substr(entry_name.find_last_of("."), suffix.length()) == suffix
                  )
				{
					// try to load the plugin
                    std::string pluginpath = std::string(plugindir) + "/" + entry_name;
                    DynLib* lib = new DynLib(pluginpath);
					if(lib->isValid()) {
						// if it is valid load the plugin
						Plugin* plugin = new Plugin(lib);
						impl->pluginlist.push_back(plugin);
                        std::string configFileName = "";
                        if (entry_name.find_last_of(".") != std::string::npos)
                        {
                            configFileName = std::string(plugindir) + "/" + entry_name.substr(0,entry_name.find_last_of(".")) + ".cfg";
                        }

                        Logger::out(0) << "Found " << getPluginTypeString(plugin->getInfo().type) << " " << plugin->getInfo().name << " (" << pluginpath << ")" << std::endl;
                        loadPlugin(plugin,platformSupport,configFileName.c_str());
					}
					else
						delete lib;
				}
			}
			// close the directory handle
			closedir(dir);
		}
		else
		{
			status = DIRECTORY_NOT_FOUND;
		}

		// forward the error message
		sendError(status);
		return status;
	}

    std::string Host::getPluginTypeString(int pluginType)
    {
        switch (pluginType)
        {
            case OPI_PROPAGATOR_PLUGIN:
                return "propagator plugin";

            case OPI_PROPAGATOR_MODULE_PLUGIN:
                return "propagator module";

            case OPI_PROPAGATOR_INTEGRATOR_PLUGIN:
                return "integrator plugin";

            case OPI_DISTANCE_QUERY_PLUGIN:
                return "distance query plugin";

            case OPI_COLLISION_DETECTION_PLUGIN:
                return "collision detection plugin";

            case OPI_COLLISION_HANDLING_PLUGIN:
                return "collision handling plugin";

            default:
                return "unknown plugin";
        };
    }

    bool Host::pluginSupported(Module *plugin, gpuPlatform platform)
    {
        bool support = false;
        int pluginMajor = plugin->minimumOPIVersionRequired();
        int pluginMinor = plugin->minorOPIVersionRequired();

        if (pluginMajor <= 0)
		{
			Logger::out(0) << "Warning: " << plugin->getName() << " has no minimum OPI version set. "
				<< "It may not function correctly and should be removed from the plugin folder." << std::endl;
			support = true;
		}
        else if (pluginMajor < OPI_API_VERSION_MAJOR)
        {
            Logger::out(0) << "Warning: " << plugin->getName() << " was made for a previous version of OPI. "
                      << "It may not function correctly and should be removed from the plugin folder." << std::endl;
			support = true;
        }
        else if (pluginMajor > OPI_API_VERSION_MAJOR)
        {
            Logger::out(0) << plugin->getName() << ": Skipped because it needs at least OPI version "
                      << pluginMajor << "." << pluginMinor << "." << std::endl;
            support = false;
        }
        else if (pluginMinor > OPI_API_VERSION_MINOR)
        {
            Logger::out(0) << plugin->getName() << ": Skipped because it needs at least OPI version "
                      << pluginMajor << "." << pluginMinor << "." << std::endl;
            support = false;
        }
        else if (plugin->requiresCUDA() <= 0 && plugin->requiresOpenCL() <= 0) {
            // no GPU support required; load plugin
            support = true;
        }
        else if (plugin->requiresCUDA() > 0) {
            if (platform != PLATFORM_CUDA) {
                Logger::out(0) << plugin->getName()
                << ": Skipped - no CUDA support available." << std::endl;
                support = false;
            }
            else if (getCurrentCudaDeviceCapability() <= 0) {
                Logger::out(0) << "[OPI] Warning: Cannot determine CUDA compute capability of selected device "
                    << "(perhaps no CUDA device was selected?). " << std::endl;
                Logger::out(0) << "[OPI] Propagator " << plugin->getName() << " requires at least "
                    << plugin->requiresCUDA() << ".x - "
                    << "otherwise propagation might fail." << std::endl;
                support = true;
            }
            else if (plugin->requiresCUDA() <= getCurrentCudaDeviceCapability()) {
                support = true;
            }
            else {
                Logger::out(0) << plugin->getName()
                << ": Skipped - Compute capability too low (" <<
                getCurrentCudaDeviceCapability() << ", needs at least" <<
                plugin->requiresCUDA() << ")" << std::endl;
                support = false;
            }
        }
        else if (plugin->requiresOpenCL() > 0) {
            if (platform == PLATFORM_OPENCL) {
                support = true;
            }
            else {
                Logger::out(0) << plugin->getName()
                << ": Skipped - no OpenCL support available." << std::endl;
                support = false;
            }
        }
        return support;
    }

    void Host::loadPlugin(Plugin *plugin, gpuPlatform platform, const char* configfile)
	{
        // we have a functional plugin loaded
        // now create the correct type for it
        switch(plugin->getInfo().type)
        {
        // propagator plugin
        case OPI_PROPAGATOR_PLUGIN:
        {
            Propagator* propagator = 0;
            // this plugin uses the cpp interface
            if(plugin->getInfo().cppPlugin) {
                pluginPropagatorFunction proc_create = (pluginPropagatorFunction)plugin->getHandle()->loadFunction("OPI_Plugin_createPropagator");
                if(proc_create)
                    propagator = proc_create(this);
            }
            // c plugin interface
            else
                propagator = new PropagatorPlugin(plugin);
            // if the propagator was created, add it to the list

            if(propagator && pluginSupported(propagator, platform)) {
                propagator->loadConfigFile(configfile);
                addPropagator(propagator);
            }
            break;
        }
            // a distance query plugin
        case OPI_DISTANCE_QUERY_PLUGIN:
        {
            DistanceQuery* query = 0;
            // this plugin uses the cpp interface
            if(plugin->getInfo().cppPlugin) {
                pluginDistanceQueryFunction proc_create = (pluginDistanceQueryFunction)plugin->getHandle()->loadFunction("OPI_Plugin_createDistanceQuery");
                if(proc_create)
                    query = proc_create(this);
            }
            // if the query plugin is valid, add it to the list
            if(query && pluginSupported(query, platform))
                addDistanceQuery(query);
            break;
        }
            // a collison detection plugin
        case OPI_COLLISION_DETECTION_PLUGIN:
        {
            CollisionDetection* cppplugin = 0;
            // this plugin uses the cpp interface
            if(plugin->getInfo().cppPlugin) {
                pluginCollisionDetectionFunction proc_create = (pluginCollisionDetectionFunction)plugin->getHandle()->loadFunction("OPI_Plugin_createCollisionDetection");
                if(proc_create)
                    cppplugin = proc_create(this);
            }
            // if the plugin is valid, add it to the corresponding list
            if(cppplugin && pluginSupported(cppplugin, platform))
                addCollisionDetection(cppplugin);
            break;
        }
            // unknown/ not implemented plugin types
        default:
            Logger::out(0) << "Skipping unknown plugin " << plugin->getInfo().name << "." << std::endl;
        }

    }

    const char* Host::getPluginPath() const
    {
        return impl->pluginPath.c_str();
    }

    Propagator* Host::getPropagator(const char* name) const
	{        
		for(size_t i = 0; i < impl->propagagorlist.size(); ++i)
		{
            if(strcmp(impl->propagagorlist[i]->getName(),name) == 0)
				return impl->propagagorlist[i];
		}
		return 0;
	}

	Propagator* Host::getPropagator(int index) const
	{
		if((index < 0)||(index >= static_cast<int>(impl->propagagorlist.size())))
		{
			sendError(INDEX_RANGE);
			return 0;
		}
		return impl->propagagorlist[index];
	}

	int Host::getPropagatorCount() const
	{
		return static_cast<int>(impl->propagagorlist.size());
	}

	int Host::getCudaDeviceCount() const
	{
		if(!impl->gpuSupport)
			return 0;
		return impl->gpuSupport->getDeviceCount();
	}
	
	int Host::selectCudaDevice(int deviceNumber) const
	{
		if(!impl->gpuSupport) {
			return -1;
		}
		else {
			impl->gpuSupport->selectDevice(deviceNumber);
			return 0;
		}
	}

	std::string Host::getCurrentCudaDeviceName() const
	{
		if(!impl->gpuSupport) {
			return std::string("No CUDA device available.");
		}
		return impl->gpuSupport->getCurrentDeviceName();
	}

	int Host::getCurrentCudaDeviceCapability() const
	{
		if(!impl->gpuSupport) {
			return 0;
		}
		return impl->gpuSupport->getCurrentDeviceCapability();
	}

	void Host::addPropagator(Propagator *propagator)
	{
		propagator->setHost(this);
		impl->propagagorlist.push_back(propagator);
	}

	void Host::addDistanceQuery(DistanceQuery *query)
	{
		query->setHost(this);
		impl->querylist.push_back(query);
	}

	int Host::getDistanceQueryCount() const
	{
		return impl->querylist.size();
	}

	void Host::addCollisionDetection(CollisionDetection *module)
	{
		module->setHost(this);
		impl->detectionlist.push_back(module);
	}

    CollisionDetection* Host::getCollisionDetection(const char* name) const
	{
		for(size_t i = 0; i < impl->detectionlist.size(); ++i)
		{
            if(strcmp(impl->detectionlist[i]->getName(), name) == 0)
				return impl->detectionlist[i];
		}
		return 0;
	}

	CollisionDetection* Host::getCollisionDetection(int index) const
	{
		if((index < 0)||(index >= static_cast<int>(impl->detectionlist.size())))
		{
			sendError(INDEX_RANGE);
			return 0;
		}
		return impl->detectionlist[index];
	}

	int Host::getCollisionDetectionCount() const
	{
		return impl->detectionlist.size();
	}

	bool Host::hasCUDASupport() const
	{
		if(impl->gpuSupport)
			return getCudaDeviceCount() > 0;
		return false;
	}

    DistanceQuery* Host::getDistanceQuery(const char* name) const
	{
		for(size_t i = 0; i < impl->querylist.size(); ++i)
		{
            if(strcmp(impl->querylist[i]->getName(),name) == 0)
				return impl->querylist[i];
		}
		return 0;
	}

	DistanceQuery* Host::getDistanceQuery(int index) const
	{
		if((index < 0)||(index >= static_cast<int>(impl->querylist.size())))
		{
			sendError(INDEX_RANGE);
			return 0;
		}
		return impl->querylist[index];
	}

    CustomPropagator* Host::createCustomPropagator(const char* name)
	{
		CustomPropagator* prop = new CustomPropagator(name);
		addPropagator(prop);
		return prop;
	}

	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	GpuSupport* Host::getGPUSupport() const
	{
		return impl->gpuSupport;
	}

	void Host::sendError(ErrorCode code) const
	{
		if(code != SUCCESS)
		{
			if(impl->errorCallback)
				impl->errorCallback((void*)(this), code, impl->errorCallbackParameter);
			impl->lastError = code;
		}
	}

	cudaDeviceProp* Host::getCUDAProperties(int device) const
	{
        if(hasCUDASupport())
			return impl->gpuSupport->getDeviceProperties(device);
		return 0;
	}

	/**
	 * \endcond
	 */
}
