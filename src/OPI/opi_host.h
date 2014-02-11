/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
#ifndef OPI_HOST_CPP_H
#define OPI_HOST_CPP_H
#include <string>
#include <vector>
#include "opi_common.h"
#include "opi_error.h"
#include "opi_pimpl_helper.h"

struct cudaDeviceProp;

namespace OPI
{
	class CustomPropagator;
	class Propagator;
	class PerturbationModule;
	class PropagatorIntegrator;
	class DistanceQuery;
	class Plugin;
	class CudaSupport;
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
	class OPI_API_EXPORT Host
	{
		public:
			Host();
			~Host();

			//! Check whether CUDA is supported on the current hardware.
			bool hasCUDASupport() const;

			//! Load all plugins found in the given directory.
			/** A plugin can be a Propagator, CustomPropagator, PerturbationModule, PropagatorIntegrator,
			 * DistanceQuery, CollisionDetection (including the C and Fortran equivalents thereof),
			 * the CUDA support plugin supplied by OPI, or any other shared object that implements the
			 * Module interface.
			 * \returns an ErrorCode containing information on any errors that occurred during the operation.
			 */
			ErrorCode loadPlugins(const std::string& plugindir);

			//! Sets an error callback for this host.
			void setErrorCallback(OPI_ErrorCallback callback, void* privatedata);

			//! Returns the number of available cuda devices
			int getCudaDeviceCount() const;

			//! Get a Propagator by index.
			/** After loading the available plugins this function can
			 * be used to get the Propagator with the given index. Indices are assigned in the order
			 * in which the plugins are loaded, starting from zero. A common use case for this function
			 * is to generate a list of available Propagators and let the user select one of them by
			 * its name.
			 * \see Host::loadPlugins
			 * \returns an instance of the Propagator with the given name; NULL if no such Propagator exists.
			 */
			Propagator* getPropagator(int index) const;

			//! Get a specific propagator by name.
			/** After loading the available plugins use this function
			 * to retrieve a specific Propagator with a known designator.
			 * \see Host::loadPlugins
			 * \returns an instance of the Propagator with the given name; NULL if no such Propagator exists.
			 */
			Propagator* getPropagator(const std::string& name) const;

			//! Returns the number of propagators.
			/** After loading the plugins, this function returns the number of valid Propagators that
			 * were found among them.
			 * \see Host::loadPlugins
			 * \returns the number of Propagator plugins available; zero if none were found.
			 */
			int getPropagatorCount() const;

			//! Adds and registers a Propagator which is not implemented by a plugin (C++-API only).
			/** If both Propagator and Host are written in C++, this function can be used to
			 * add a statically compiled Propagator to the Host's list of available propagators.
			 * Useful if Host and Propagator should be the same application (which would partly
			 * defeat the purpose of this API, but hey - it's your work :-P).
			 */
			void addPropagator(Propagator* propagator);

			//! Adds an empty CustomPropagator with the given name to the list of available Propagators.
			/** A CustomPropagator works exactly like a Propagator, but is put together from
			 * components (Perturbation Modules and Integrators) chosen by the Host.
			 * \see CustomPropagator, PerturbationModule, PropagatorIntegrator
			 * \returns a new instance of a CustomPropagator.
			 */
			CustomPropagator* createCustomPropagator(const std::string& name);

			//! Find a propagator module by name, returns 0 (null pointer) if not found
			PerturbationModule* getPerturbationModule(const std::string& name) const;
			//! Find a propagator module by index, returns 0 (null pointer) if not found
			PerturbationModule* getPerturbationModule(int index) const;
			//! Returns the number of known modules
			int getPerturbationModuleCount() const;

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
			Host(const Host& other);
			//! Load a specific plugin
			void loadPlugin(Plugin* plugin);
			Pimpl<HostImpl> impl;
	};
}

#endif
