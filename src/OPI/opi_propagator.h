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

            void loadConfigFile();
            void loadConfigFile(const std::string& filename);

            /**
             * @brief propagate Starts the propagation for the given time frame.
             *
             * This function calls the propagator and performs propagation. The resulting
             * orbit and position/velocity vectors (if supported) are written back to the
             * given Population. The propagation result shall reflect the state of the
             * population at the point in time defined by julian_day + dt seconds.
             * @param data The population to be propagated
             * @param julian_day The base date in Julian date format
             * @param dt The time step, in seconds, from last propagation.
             * @return OPI::SUCCESS if propagation was successful, or other error code.
             */
			ErrorCode propagate(Population& data, double julian_day, float dt );
			//! Starts the index-based propagation for the given time frame
			ErrorCode propagate(Population& data, IndexList& indices, double julian_day, float dt);
            //! Starts propagation with individual times for each object
            ErrorCode propagate(Population& data, double* julian_days, int length, float dt);

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

            //! Return the reference frame for the cartesian state vectors
            virtual ReferenceFrame referenceFrame();

            /**
             * @brief requiresCUDA Check whether this propagator requires CUDA to function.
             * @return 0 if CUDA is not required, or the major number of
             *  the minimum required compute capability.
             */
			virtual int requiresCUDA();

			//! Check whether this propagator requires OpenCL to function.
			/** Set to zero if OpenCL is not required, otherwise set this to the major number of
			*  the required compute capability.
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
			virtual ErrorCode runPropagation(Population& data, double julian_day, float dt ) = 0;
			//! Override this to implement an index-based propagation
			//! The C Namespace equivalent for this function is OPI_Plugin_propagateIndexed
			virtual ErrorCode runIndexedPropagation(Population& data, IndexList& indices, double julian_day, float dt );
            //! Override this to implement propagation with individual times.
            //! OPI will make sure that the julian_days vector length matches that of the Population.
            virtual ErrorCode runMultiTimePropagation(Population& data, double* julian_days, float dt);
            //! Variable to hold the appropriate name for the config file.
            std::string configFileName;

		private:
            //! Auxiliary function for loadConfig
            std::vector<std::string> tokenize(std::string line, std::string delimiter);
			Pimpl<PropagatorImpl> data;

	};
}

#endif
