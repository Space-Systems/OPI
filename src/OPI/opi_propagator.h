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
#ifndef OPI_PROPAGATOR_H
#define OPI_PROPAGATOR_H
#include "opi_common.h"
#include "opi_population.h"
#include "opi_error.h"
#include "opi_module.h"
#include "opi_pimpl_helper.h"
#include <string>
#include <vector>
namespace OPI
{
	class Population;
	class IndexList;
	class PerturbationModule;

	//! Contains the propagation implementation data
	class PropagatorImpl;

	/*!
	 * \brief This class represents a propagator loadable by a Host, usually as a shared object.
	 *
	 * \ingroup CPP_API_GROUP
	 * The Propagator takes as input a list of orbital objects and calculates their position at a given
	 * time. It is implemented as a Module loaded and managed by a Host application that uses its
	 * results for its specific purpose. The interface provides methods to initialize and configure the
	 * Propagator, and to forward Object and time data.
	 *
	 * A Propagator implementation has to implement the runPropagation function and can an optional
	 * implementation for runIndexPropagation
	 * \see Module, Host
	 */
	class OPI_API_EXPORT Propagator:
			public Module
	{
		public:
			Propagator();
			virtual ~Propagator();

            /**
             * @brief loadConfigFile Attempts to load the standard configuration file.
             *
             * The standard configuration file resides in the same directory as the plugin
             * with a .cfg suffix instead of the platform's library extension (.dll, .so,
             * .dynlib). This file is automatically loaded by the host on initialization using
             * the second variant of this function below.
             * It is recommended for plugin authors to call this function on runDisable()
             * as part of resetting the propagator to its default state.
             */
            void loadConfigFile();

            /**
             * @brief loadConfigFile Attempts to load a config file. Called by the host upon initialization.
             *
             * This function will automatically be called by OPI (and, in most cases, should
             * only ever be called by OPI) when the plugin is first loaded.
             * The given config file name will be stored in the propagator. Plugin authors
             * should use the above variant of this function when resetting the propagator.
             * @param filename The name of the config file to load.
             */
            void loadConfigFile(const std::string& filename);

            /**
             * @brief propagate Starts the propagation for the given time step.
             *
             * This function calls the propagator and performs propagation via the
             * runPropagation() function that must be implemented by the plugin. The resulting
             * orbit and position/velocity vectors (if supported) are written back to the
             * given Population. The propagation result shall reflect the state of the
             * Population at the point in time defined by julian_day + dt seconds.
             * @param data The Population to be propagated.
             * @param julian_day The base date in Julian date format.
             * @param dt The time step, in seconds, from last propagation.
             * @return OPI::SUCCESS if propagation was successful, or other error code.
             */
            ErrorCode propagate(Population& data, double julian_day, double dt);

            /**
             * @brief propagate Starts the index-based propagation for the given time step.
             *
             * Like the propagate() function above, but
             * only those Population elements that appear in the given IndexList will be
             * propagated. This function will call the runIndexedPropagation() function that
             * should be implemented by the plugin. If the plugin returns NOT_IMPLEMENTED, the
             * operation will still be performed by calling the runPropagation() function on
             * individual elements which is likely to be very inefficient.
             * @param data The Population to be propagated.
             * @param indices An IndexList containing the indices of the Population elements that
             * should be propagated.
             * @param julian_day The base date in Julian date format.
             * @param dt The time step, in seconds, from last propagation.
             * @return OPI::SUCCESS if propagation was successful; OPI::NOT_IMPLEMENTED if propagation
             * was performed with OPI's inherent method (which should still give you valid results); or
             * any other error code returned by the plugin.
             */
            ErrorCode propagate(Population& data, IndexList& indices, double julian_day, double dt);

            /**
             * @brief propagate Starts propagation with individual times for each object.
             *
             * Like the propagate) function above, but every object receives an individual base date.
             * This is useful e.g. when doing fine-grained conjunction analysis between two
             * regular time steps. This function will call runMultiTimePropagation() which should be
             * implemented by the plugin. If the plugin returns NOT_IMPLEMENTED, the operation will
             * be performed by instead calling runPropagation() on individual objects with individual
             * times. While this should yield valid results, it is likely to be very inefficient.
             * @param data The Population to be propagated.
             * @param julian_days An array of Julian dates, one for each element in the Population.
             * @param length The length of the julian_days array. Must be the same size as the Population's.
             * @param dt The time step, in seconds, from last propagation.
             * @return OPI::SUCCESS if propagation was successful; OPI::NOT_IMPLEMENTED if propagation
             * was performed with OPI's inherent method (which should still give you valid results); or
             * any other ErrorCode returned by the plugin.
             */
            ErrorCode propagate(Population& data, double* julian_days, int length, double dt);

			//! Assigns a module to this propagator
			/**
			 * It depends on the used Propagator if the assigned modules will be used
			 */
			PerturbationModule* assignPerturbationModule(const std::string& name);
			//! Returns true if the propagator is able to use Perturbation Modules
			bool usesModules() const;

			//! Returns the assigned Perturbation modules
			PerturbationModule* getPerturbationModule(int index);

			//! Returns the number of assigned Perturbation modules
			int getPerturbationModuleCount() const;

			//! Check if this propagator is able to propagate backwards
			virtual bool backwardPropagation();
	
			//! Check if this propagator supports generation of cartesian state vectors
			virtual bool cartesianCoordinates();

            /**
             * @brief referenceFrame Return the reference frame for the cartesian state vectors.
             *
             * Set by the propagator to specify in which reference frame the state vectors (position,
             * velocity, and acceleration) are given. Defaults to REF_NONE if cartesianCoordinates()
             * returns false, or REF_UNSPECIFIED otherwise. Set to REF_UNLISTED if the reference frame
             * is non-standard or otherwise does not appear in the ReferenceFrame enum. Since frames come
             * in many different flavours always consult the plugin's documentation for specifics.
             * @return A value of the ReferenceFrame enum matching the propagator's output.
             */
            virtual ReferenceFrame referenceFrame();

            /**
             * @brief requiresCUDA Check whether this propagator requires CUDA to function.
             * @return 0 if CUDA is not required, or the major number of
             *  the minimum required compute capability.
             */
			virtual int requiresCUDA();

            /**
             * @brief Check whether this propagator requires OpenCL to function.
             * @return 0 if OpenCL is not required, otherwise set this to the major number of
             * the required OpenCL version.
             */
			virtual int requiresOpenCL();

            /**
             * @brief minimumOPIVersionRequired Returns the minimum OPI API level that this propagator requires.
             *
             * API level is equal to OPI's major version number.
             * @return An integer representing the minimum API level required.
             */
            virtual int minimumOPIVersionRequired();           

		protected:
			//! Defines that this propagator (can) use Perturbation Modules
			void useModules();
			//! The actual propagation implementation
			//! The C Namespace equivalent for this function is OPI_Plugin_propagate
            virtual ErrorCode runPropagation(Population& data, double julian_day, double dt) = 0;
			//! Override this to implement an index-based propagation
			//! The C Namespace equivalent for this function is OPI_Plugin_propagateIndexed
            virtual ErrorCode runIndexedPropagation(Population& data, IndexList& indices, double julian_day, double dt);
            //! Override this to implement propagation with individual times.
            //! OPI will make sure that the julian_days vector length matches that of the Population.
            virtual ErrorCode runMultiTimePropagation(Population& data, double* julian_days, int length, double dt);
            //! Variable to hold the appropriate name for the config file.
            std::string configFileName;

		private:
            //! Auxiliary function for loadConfig
            std::vector<std::string> tokenize(std::string line, std::string delimiter);
			Pimpl<PropagatorImpl> data;

	};
}

#endif
