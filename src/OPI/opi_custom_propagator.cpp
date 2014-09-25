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
#include "opi_custom_propagator.h"

namespace OPI
{
	struct CustomPropagatorImpl
	{
		std::vector<PerturbationModule*> modules;
		PropagatorIntegrator* integrator;
	};

	CustomPropagator::CustomPropagator(const std::string &name)
	{
		setName(name);
		impl->integrator = 0;
	}

	CustomPropagator::~CustomPropagator()
	{

	}

	void CustomPropagator::addModule(PerturbationModule *module)
	{
		impl->modules.push_back(module);
	}

	void CustomPropagator::setIntegrator(PropagatorIntegrator *integrator)
	{
		impl->integrator = integrator;
	}

	ErrorCode CustomPropagator::runPropagation(ObjectData& data, double julian_day, float dt )
	{
		return NO_ERROR;
	}

	bool CustomPropagator::requiresCUDA()
	{
		return false;
	}
}
