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
#ifndef OPI_HOST_MODULED_PROPAGATOR_H
#define OPI_HOST_MODULED_PROPAGATOR_H

#include "opi_propagator.h"
#include <vector>
namespace OPI
{
	class PerturbationModule;
	class PropagatorIntegrator;

	struct CustomPropagatorImpl;

	//! \brief This class represents a propagator which can be composed from different perturbation modules and an integrator at runtime.
	//! \ingroup CPP_API_GROUP
	class CustomPropagator:
			public Propagator
	{
		public:
			//! Creates a new custom propagator with the specified name
			OPI_API_EXPORT CustomPropagator(const std::string& name);
			OPI_API_EXPORT ~CustomPropagator();
			/// Adds a module to this propagator
			OPI_API_EXPORT void addModule(PerturbationModule* module);
			/// Sets the integrator for this propagator
			OPI_API_EXPORT void setIntegrator(PropagatorIntegrator* integrator);

		protected:
			/// Override the propagation method
            virtual ErrorCode runPropagation(Population& population, double julian_day, double dt);
			virtual int requiresCUDA();

		private:
			Pimpl<CustomPropagatorImpl> impl;
	};
}

#endif
