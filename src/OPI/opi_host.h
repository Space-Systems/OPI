#ifndef OPI_HOST_CPP_H
#define OPI_HOST_CPP_H
#include <string>
#include <vector>
#include "opi_common.h"
#include "opi_error.h"
#include "opi_pimpl_helper.h"
#include "opi_module.h"

struct cudaDeviceProp;

namespace OPI
{
	class CustomPropagator;
	class Propagator;
	class PerturbationModule;
	class PropagatorIntegrator;
	class DistanceQuery;
	class Plugin;
	class GpuSupport;
	class DynLib;
	class CollisionDetection;

	//! Internal implementation data for the Host
	class HostImpl;

	/*!
	 * \brief The Host loads and manages one or multiple Propagator Plugins.
	 *
	 * \ingroup CPP_API_GROUP
	 * Usually, the Host program is your main application (or part thereof) that serves a specific purpose
	 * for which orbital propagation is required; for example, a re-entry simulation, an animated
	 * visualization of space debris, or a small tool that simply compares different propagators.
	 * When using C++, an instance of the Host class would be the part of your application responsible
	 * for orbital propagation. The Host keeps a list of available propagators (usually loaded from
	 * shared objects) and provides access to them.
	 */
	class Host
	{
		public:
            OPI_API_EXPORT Host();
			OPI_API_EXPORT ~Host();
	
			enum gpuPlatform {
				PLATFORM_NONE,
				PLATFORM_CUDA,
				PLATFORM_OPENCL
			};

			//! Let OPI and supported propagators log their output to a file.
			/** This will redirect all output from OPI and from propagators that use the OPI::Logger
			 * for debugging output. You can use the propagator's supportsOPILogger() function to
			 * check whether or not it uses the logger.
			 * \param logfileName Name of the log file, or "" to revert back to stdout.
			 * \param append Set to true to append to an existing file instead of overwriting it.
			 */
			OPI_API_EXPORT void logToFile(const char* logfileName, bool append = false);

			//! Set the desired amount of detail for the log output.
			/** How this is used depends largely on the implementation in the propagator, but
			 * in general, verbose levels are:
             * -1 : No message output at all.
		     *  0 : Default level: Regular output and errors.
		     *  1 : Warnings.
		     *  2 : Status messages.
		     *  3 : Detailed status and debug messages.
		     *  4 : Very detailed status and debug messages.
			 */
			OPI_API_EXPORT void setVerboseLevel(int level);


			//! Check whether CUDA is supported on the current hardware.
			OPI_API_EXPORT bool hasCUDASupport() const;

			//! Load all plugins found in the given directory.
			/** A plugin can be a Propagator, CustomPropagator, PerturbationModule, PropagatorIntegrator,
			 * DistanceQuery, CollisionDetection (including the C and Fortran equivalents thereof),
			 * or any other shared object that implements the Module interface.
			 * The parameter platformSupport states whether support for CUDA (default) or OpenCL should be loaded.
			 * \returns an ErrorCode containing information on any errors that occurred during the operation.
			 */
            OPI_API_EXPORT ErrorCode loadPlugins(const char* plugindir, gpuPlatform platformSupport = PLATFORM_NONE, int platformNumber = 0, int deviceNumber = 0);

			//! Sets an error callback for this host.
			OPI_API_EXPORT void setErrorCallback(OPI_ErrorCallback callback, void* privatedata);

			//! Returns the number of available CUDA devices
			OPI_API_EXPORT int getCudaDeviceCount() const;

			//! Selects the CUDA device to be used by the plugin.
			/** The default device is zero.
			 * \returns -1 if no CUDA devices are present, zero otherwise.
			 */	
			OPI_API_EXPORT int selectCudaDevice(int deviceNumber) const;

			//! Returns the name of the currently selected CUDA device.
			OPI_API_EXPORT std::string getCurrentCudaDeviceName() const;

			//! Returns the major capability number of the currently selected CUDA device.
			OPI_API_EXPORT int getCurrentCudaDeviceCapability() const;

            //! Returns the current plugin path.
            OPI_API_EXPORT const char* getPluginPath() const;

			//! Get a Propagator by index.
			/** After loading the available plugins this function can
			 * be used to get the Propagator with the given index. Indices are assigned in the order
			 * in which the plugins are loaded, starting from zero. A common use case for this function
			 * is to generate a list of available Propagators and let the user select one of them by
			 * its name.
			 * \see Host::loadPlugins
			 * \returns an instance of the Propagator with the given name; NULL if no such Propagator exists.
			 */
			OPI_API_EXPORT Propagator* getPropagator(int index) const;

			//! Get a specific propagator by name.
			/** After loading the available plugins use this function
			 * to retrieve a specific Propagator with a known designator.
			 * \see Host::loadPlugins
			 * \returns an instance of the Propagator with the given name; NULL if no such Propagator exists.
			 */
            OPI_API_EXPORT Propagator* getPropagator(const char* name) const;

			//! Returns the number of propagators.
			/** After loading the plugins, this function returns the number of valid Propagators that
			 * were found among them.
			 * \see Host::loadPlugins
			 * \returns the number of Propagator plugins available; zero if none were found.
			 */
			OPI_API_EXPORT int getPropagatorCount() const;

			//! Adds and registers a Propagator which is not implemented by a plugin (C++-API only).
			/** If both Propagator and Host are written in C++, this function can be used to
			 * add a statically compiled Propagator to the Host's list of available propagators.
			 * Useful if Host and Propagator should be the same application (which would partly
			 * defeat the purpose of this API, but hey - it's your work :-P).
			 */
			OPI_API_EXPORT void addPropagator(Propagator* propagator);

			//! Adds an empty CustomPropagator with the given name to the list of available Propagators.
			/** A CustomPropagator works exactly like a Propagator, but is put together from
			 * components (Perturbation Modules and Integrators) chosen by the Host.
			 * \see CustomPropagator, PerturbationModule, PropagatorIntegrator
			 * \returns a new instance of a CustomPropagator.
			 */
            OPI_API_EXPORT CustomPropagator* createCustomPropagator(const char* name);

            /* NOT YET IMPLEMENTED
			//! Find a propagator module by name, returns 0 (null pointer) if not found
            PerturbationModule* getPerturbationModule(const char* name) const;
			//! Find a propagator module by index, returns 0 (null pointer) if not found
			PerturbationModule* getPerturbationModule(int index) const;
			//! Returns the number of known modules
			int getPerturbationModuleCount() const;

			//! Find a propagator integrator by name, returns 0 (null pointer) if not found
            PropagatorIntegrator* getPropagatorIntegrator(const char* name) const;
			//! Find a propagator integrator by index, returns 0 (null pointer) if not found
			PropagatorIntegrator* getPropagatorIntegrator(int index) const;

			//! Returns the number of known Intergrators
			int getPropagatorIntegratorCount() const;
            */

			//! Adds and registers a Distance query which is not implemented by a plugin
			OPI_API_EXPORT void addDistanceQuery(DistanceQuery* query);
			//! Returns an distance query module by name, returns 0 (null pointer) if not found
            OPI_API_EXPORT DistanceQuery* getDistanceQuery(const char* name) const;
			//! Returns an distance query module by index, returns 0 (null pointer) if not found
			OPI_API_EXPORT DistanceQuery* getDistanceQuery(int index) const;
			//! Returns the number of known distance queries
			OPI_API_EXPORT int getDistanceQueryCount() const;

			//! Adds and registers a Collision Detection module
			OPI_API_EXPORT void addCollisionDetection(CollisionDetection* module);
			//! Find a collision detection module by name, returns 0 (null pointer) if not found
            OPI_API_EXPORT CollisionDetection* getCollisionDetection(const char* name) const;
			//! Find a collision detection module by index, returns 0 (null pointer) if not found
			OPI_API_EXPORT CollisionDetection* getCollisionDetection(int index) const;
			//! Returns the number of known collision detection modules
			OPI_API_EXPORT int getCollisionDetectionCount() const;


			//! Returns the code of the last occured Error of this host or its plugins
			OPI_API_EXPORT ErrorCode getLastError() const;

			//! \cond INTERNAL_DOCUMENTATION

			//! Returns the CUDA Support object
			OPI_API_EXPORT GpuSupport* getGPUSupport() const;
			//! Returns cuda device properties
			OPI_API_EXPORT cudaDeviceProp* getCUDAProperties(int device = 0) const;

			//! Sends an error through the registered callback
			OPI_API_EXPORT void sendError(ErrorCode code) const;
			//! \endcond
		private:
			Host(const Host& other);
			//! Load a specific plugin
            void loadPlugin(Plugin* plugin, gpuPlatform platform, const char* configfile);
            bool pluginSupported(Module *plugin, gpuPlatform platform);
            std::string getPluginTypeString(int pluginType);
			Pimpl<HostImpl> impl;
	};
}

#endif
