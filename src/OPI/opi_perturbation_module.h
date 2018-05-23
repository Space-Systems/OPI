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
#ifndef OPI_PERTURBATION_MODULE_H
#define OPI_PERTURBATION_MODULE_H

#include "opi_common.h"
#include "opi_module.h"
#include "opi_perturbations.h"
#include "opi_error.h"
#include "opi_pimpl_helper.h"
namespace OPI
{
	class Population;
    class Perturbations;

	//! Contains the module implementation data
	class PerturbationModuleImpl;

	/*!
     * \brief This class represents a perturbation module which can be used by a Propagator
	 *
	 * \ingroup CPP_API_GROUP
	 * \see Module, Host
	 */
	class PerturbationModule: public Module
	{
		public:
			OPI_API_EXPORT PerturbationModule();
			OPI_API_EXPORT virtual ~PerturbationModule();
            //! Calculates the Perturbation for the passed dataset
			/**
             * The calculated perturbation forces will be added to the values present in delta
			 */
			OPI_API_EXPORT ErrorCode calculate(Population& data, Perturbations& delta, double julian_day, double dt);

		protected:
            virtual ErrorCode runCalculation(Population& data, Perturbations& delta, double julian_day, double dt );

		private:
			Pimpl<PerturbationModuleImpl> impl;
	};
}

#endif
