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
	class Propagator: public Module
	{
		public:
			OPI_API_EXPORT Propagator();
			OPI_API_EXPORT virtual ~Propagator();

            /**
             * @brief propagate Starts the propagation for the given time step.
             *
             * This function calls the propagator and performs propagation via the
             * runPropagation() function that must be implemented by the plugin. The resulting
             * orbit and position/velocity vectors (if supported) are written back to the
             * given Population. The propagation result shall reflect the state of the
             * Population at the point in time defined by julian_day + dt seconds.
             * @param population The Population to be propagated.
             * @param julian_day The base date in Julian date format. Ignored when mode is set to
             * individual epochs.
             * @param dt The time step, in seconds, from last propagation.
             * @param mode Sets the propagation mode to single epoch (default) or individual epochs.
             * In single epoch mode, the propagator assumes that all objects are at the same epoch
             * given with the julian_day parameter. In individual epoch mode, that parameter is
             * ignored and the objects' individual current_epoch parameter is used instead.
             * The plugin shall return NOT_IMPLEMENTED if propagation with individual epochs
             * is unsupported.
             * @param indices An IndexList containing the indices of the Population elements that
             * should be propagated. Defaults to null in which case all objects are propagated.
             * The plugin shall return NOT_IMPLEMENTED if an index list is set and indexed propagation
             * is unsupported.
             * @return OPI::SUCCESS if propagation was successful, or other error code.
             */
            OPI_API_EXPORT ErrorCode propagate(Population& population, double julian_day, double dt, PropagationMode mode = MODE_SINGLE_EPOCH, IndexList* indices = nullptr);

            //! Assigns a module to this propagator (not yet implemented)
			/**
			 * It depends on the used Propagator if the assigned modules will be used
			 */
            //PerturbationModule* assignPerturbationModule(const char* name);
			//! Returns true if the propagator is able to use Perturbation Modules
			OPI_API_EXPORT bool usesModules() const;

			//! Returns the assigned Perturbation modules
			OPI_API_EXPORT PerturbationModule* getPerturbationModule(int index);

			//! Returns the number of assigned Perturbation modules
			OPI_API_EXPORT int getPerturbationModuleCount() const;

			//! Check if this propagator is able to propagate backwards
			OPI_API_EXPORT virtual bool backwardPropagation();
	
			//! Check if this propagator supports generation of cartesian state vectors
			OPI_API_EXPORT virtual bool cartesianCoordinates();

            /**
             * @brief referenceFrame Return the reference frame for the cartesian state vectors.
             *
             * Set by the propagator to specify in which reference frame the state vectors (position,
             * velocity, and acceleration) are given. Defaults to REF_NONE if cartesianCoordinates()
             * returns false, or REF_UNSPECIFIED otherwise. Set to REF_UNLISTED if the reference frame
             * is non-standard or otherwise does not appear in the ReferenceFrame enum. If the propagator
             * supports multiple reference frames, this can be set to REF_MULTIPLE, and a
             * PropagatorProperty can be used to let the user select the desired option. Since frames come
             * in many different flavours always consult the plugin's documentation for specifics.
             * @return A value of the ReferenceFrame enum matching the propagator's output.
             */
			OPI_API_EXPORT virtual ReferenceFrame referenceFrame();

            /**
             * @brief covarianceType Return the setup of the covariance matrix.
             *
             * Set by the propagator to specify the meaning of the kinematic (k1-k6) and dynamic (d1-d2)
             * elements of the covariance matrix. Defaults to CV_NONE meaning the propagator does not
             * support covariances. Set to CV_STATE_VECTORS if the kinematic parameters are state vectors
             * (k1-k3 for position x/y/z and k4-k6 for velocity x/y/z). Set to CV_EQUINOCTIALS if the
             * kinematic parameters are equinoctial elements, or CV_KEPLERIAN if they are orbital elements
             * (k1-k6 for semi major axis, eccentricity, inclination, raan, argument of perigee and mean
             * anomaly). The options ending in _NO_DYNAMICS state that the dynamic parameters are unused.
             */
            OPI_API_EXPORT virtual CovarianceType covarianceType();

            /**
             * @brief Initializes a Population from a file or path.
             *
             * Some propagators (such as the well-known SGP4 propagator) are designed to work with
             * very specific data formats. This function can be used by the plugin author to
             * provide a method to load a Population from a file or directory. Additional configuration
             * options can be provided via PropagatorProperties if required.
             * @param population A pointer to an empty Population that will hold the data.
             * @param filename The name of the file or path that holds the population data.
             * @return OPI::SUCCESS if operation was successful, or other error code. Defaults to OPI::NOT_IMPLEMENTED.
             */
            OPI_API_EXPORT virtual ErrorCode loadPopulation(Population& population, const char* filename);

            /**
             * @brief Align a population to a common epoch.
             *
             * This function allows a population to be aligned to a common epoch. The epoch is chosen automatically
             * by analyzing the population and finding the object whose epoch is furthest ahead. All other objects
             * will then be propagated to that epoch.
             * All propagators that correctly implement both indexed propagation and individual epoch mode should be
             * capable of supporting object alignment. All objects in the population will need their current epoch
             * set to a value larger than zero for this to work.
             * @param population The population to be aligned.
             * @param dt The step size, in seconds, used for alignment.
             * @return OPI::NOT_IMPLEMENTED if the propagator does not support the required functions, OPI::INVALID_VALUE
             * if the population does not have all required fields filled out, OPI::SUCCESS if the propagator returns
             * no errors.
             */
            OPI_API_EXPORT virtual OPI::ErrorCode align(OPI::Population& population, double dt);

		protected:
			//! Defines that this propagator (can) use Perturbation Modules
			void useModules();
			//! The actual propagation implementation
			//! The C Namespace equivalent for this function is OPI_Plugin_propagate
            virtual ErrorCode runPropagation(Population& population, double julian_day, double dt, PropagationMode mode = MODE_SINGLE_EPOCH, IndexList* indices = nullptr) = 0;

		private:
			Pimpl<PropagatorImpl> data;

	};
}

#endif
