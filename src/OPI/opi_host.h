#ifndef OPI_HOST_CPP_H
#define OPI_HOST_CPP_H
#include <string>
#include <vector>
#include "opi_common.h"
#include "opi_error.h"

struct cudaDeviceProp;

namespace OPI
{
	class CustomPropagator;
	class Propagator;
	class PropagatorModule;
	class PropagatorIntegrator;
	class DistanceQuery;
	class Plugin;
	class CudaSupport;
	class DynLib;
	class CollisionDetection;

	//! Internal implementation data for the Host
	class HostImpl;

	//! \brief The Host loads and manages plugins
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT Host
	{
		public:
			Host();
			~Host();

			//! Check if CUDA is supported
			bool hasCUDASupport() const;
			//! Load plugins from plugindir
			ErrorCode loadPlugins(const std::string& plugindir);

			//! Sets an error callback for this host
			void setErrorCallback(OPI_ErrorCallback callback, void* privatedata);

			//! Returns the number of available cuda devices
			int getCudaDeviceCount() const;
			//! Get a propagator by name
			Propagator* getPropagator(const std::string& name) const;
			//! Get a propagator by index
			Propagator* getPropagator(int index) const;
			//! Returns the number of propagators
			int getPropagatorCount() const;
			//! Adds and registers a Propagator which is not implemented by a plugin (cpp-api only)
			void addPropagator(Propagator* propagator);

			//! Creates a custom propagator with the passed name
			CustomPropagator* createCustomPropagator(const std::string& name);
			//! Find a propagator module by name, returns 0 (null pointer) if not found
			PropagatorModule* getPropagatorModule(const std::string& name) const;
			//! Find a propagator module by index, returns 0 (null pointer) if not found
			PropagatorModule* getPropagatorModule(int index) const;
			//! Returns the number of known modules
			int getPropagatorModuleCount() const;

			//! Find a propagator integrator by name, returns 0 (null pointer) if not found
			PropagatorIntegrator* getPropagatorIntegrator(const std::string& name) const;
			//! Find a propagator integrator by index, returns 0 (null pointer) if not found
			PropagatorIntegrator* getPropagatorIntegrator(int index) const;

			//! Returns the number of known Intergrators
			int getPropagatorIntegratorCount() const;

			//! Adds and registers a Distance query which is not implemented by a plugin
			void addDistanceQuery(DistanceQuery* query);
			//! Returns an distance query module by name, returns 0 (null pointer) if not found
			DistanceQuery* getDistanceQuery(const std::string& name) const;
			//! Returns an distance query module by index, returns 0 (null pointer) if not found
			DistanceQuery* getDistanceQuery(int index) const;
			//! Returns the number of known distance queries
			int getDistanceQueryCount() const;

			//! Adds and registers a Collision Detection module
			void addCollisionDetection(CollisionDetection* module);
			//! Find a collision detection module by name, returns 0 (null pointer) if not found
			CollisionDetection* getCollisionDetection(const std::string& name) const;
			//! Find a collision detection module by index, returns 0 (null pointer) if not found
			CollisionDetection* getCollisionDetection(int index) const;
			//! Returns the number of known collision detection modules
			int getCollisionDetectionCount() const;


			//! Returns the code of the last occured Error of this host or its plugins
			ErrorCode getLastError() const;

			//! \cond INTERNAL_DOCUMENTATION

			//! Returns the CUDA Support object
			CudaSupport* getCUDASupport() const;
			//! Returns cuda device properties
			cudaDeviceProp* getCUDAProperties(int device = 0) const;

			//! Sends an error through the registered callback
			void sendError(ErrorCode code) const;
			//! \endcond
		private:
			//! Load a specific plugin
			void loadPlugin(Plugin* plugin);
			HostImpl* impl;
	};
}

#endif
