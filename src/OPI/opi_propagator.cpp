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
#include "opi_indexlist.h"
#include <iostream>
#include <iomanip>

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

    ErrorCode Propagator::propagate(Population& population, double julian_day, double dt, PropagationMode mode, IndexList* indices)
	{
		ErrorCode status = SUCCESS;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == SUCCESS)
            status = runPropagation(population, julian_day, dt, mode, indices);
		getHost()->sendError(status);
        if (status == SUCCESS && population.getLastPropagatorName() != getName())
        {
            population.setLastPropagatorName(getName());
        }
		return status;
	}

	bool Propagator::backwardPropagation()
	{
        return false;
	}

	bool Propagator::cartesianCoordinates()
	{
		return false;
	}

    ReferenceFrame Propagator::referenceFrame()
    {
        if (cartesianCoordinates())
        {
            return REF_UNSPECIFIED;
        }
        else return REF_NONE;
    }

    CovarianceType Propagator::covarianceType()
    {
        return CV_NONE;
    }

    ErrorCode Propagator::loadPopulation(Population& population, const char* filename)
    {
        return NOT_IMPLEMENTED;
    }  

    OPI::ErrorCode Propagator::align(OPI::Population& population, double dt)
    {
        // Find the target epoch to align the population to
        const double mjd1950 = 2433282.5;
        double latestEpoch = population.getLatestEpoch();
        if (latestEpoch < mjd1950)
        {
            std::cout << "Cannot align: Current epoch must be set for all objects." << std::endl;
            return OPI::INVALID_DATA;
        }
        std::cout << "Aligning to epoch " << std::setprecision(15) << latestEpoch << std::endl;

        OPI::ErrorCode error;
        int objectsAligned = 0;
        while (objectsAligned < population.getSize())
        {
            objectsAligned = 0;
            OPI::IndexList trailingObjects(population.getHostPointer());
            for (int i=0; i<population.getSize(); i++)
            {                
                const double currentEpoch = population.getEpoch()[i].current_epoch;
                const double deltaSeconds = (latestEpoch - currentEpoch) * 86400.0;
                if (deltaSeconds < 1)
                {
                    objectsAligned++;
                }
                else if (deltaSeconds >= dt)
                {
                    trailingObjects.add(i);
                }
                else if (deltaSeconds < dt)
                {
                    // Object is close to the target epoch. Propagate individually.
                    OPI::IndexList thisObject(population.getHostPointer());
                    thisObject.add(i);
                    //std::cout << "Object " << i << " closing in. Propagating for " << deltaSeconds << " seconds." << std::endl;
                    error = runPropagation(population, 0.0, deltaSeconds, OPI::MODE_INDIVIDUAL_EPOCHS, &thisObject);
                }
            }
            if (trailingObjects.getSize() > 0) error = runPropagation(population, 0.0, dt, OPI::MODE_INDIVIDUAL_EPOCHS, &trailingObjects);
            if (error == OPI::NOT_IMPLEMENTED)
            {
                std::cout << "Cannot align: Propagator does not support the required functions." << std::endl;
                return error;
            }
            else if (error != OPI::SUCCESS)
            {
                std::cout << "Propagator returned an error during object alignment." << std::endl;
                return error;
            }
            //std::cout << objectsAligned << " objects aligned." << std::endl;
        }
        //std::cout << "Alignment complete." << std::endl;
        return OPI::SUCCESS;
    }
}
