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
#include "opi_propagator.h"
#include "opi_host.h"
#include "opi_perturbation_module.h"
#include <vector>
namespace OPI
{
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	class PropagatorImpl
	{
		public:
			PropagatorImpl():
				allowPerturbationModules(false)
			{
			}

			bool allowPerturbationModules;
			std::vector<PerturbationModule*> perturbationModules;
	};

	//! \endcond

	Propagator::Propagator()
	{
	}

	Propagator::~Propagator()
	{
	}

	void Propagator::useModules()
	{
		data->allowPerturbationModules = true;
	}

	bool Propagator::usesModules() const
	{
		return data->allowPerturbationModules;
	}

	PerturbationModule* Propagator::getPerturbationModule(int index)
	{
		return data->perturbationModules[index];
	}

	int Propagator::getPerturbationModuleCount() const
	{
		return data->perturbationModules.size();
	}

	ErrorCode Propagator::propagate(ObjectData& objectdata, float years, float seconds, float dt)
	{
		ErrorCode status = NO_ERROR;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == NO_ERROR)
			status = runPropagation(objectdata, years, seconds, dt);
		getHost()->sendError(status);
		return status;
	}

	/**
	 * If the runPropagation method for index-based propagation is not overloaded (returning OPI_NOT_IMPLEMENTED)
	 * this function will perform a normal propagation instead.
	 */
	ErrorCode Propagator::propagate(ObjectData& objectdata, IndexList& indices, float years, float seconds, float dt)
	{
		ErrorCode status = NO_ERROR;
		status = runIndexedPropagation(objectdata, indices, years, seconds, dt);
		if(status == NOT_IMPLEMENTED)
			status = propagate(objectdata, years, seconds, dt);
		getHost()->sendError(status);
		return status;
	}

	bool Propagator::backwardPropagation()
	{
		return true;
	}

	ErrorCode Propagator::runIndexedPropagation(ObjectData& data, IndexList& indices, float years, float seconds, float dt)
	{
		return NOT_IMPLEMENTED;
	}

}
