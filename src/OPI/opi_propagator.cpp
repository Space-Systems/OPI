#include "opi_propagator.h"
#include "opi_host.h"
#include "opi_perturbation_module.h"
#include "opi_indexlist.h"
#include "opi_logger.h"
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

    ErrorCode Propagator::align(Population& population, double dt, IndexList* indices, double toEpoch, bool quiet)
    {
        int loopSize = (indices ? indices->getSize() : population.getSize());
        if (loopSize > 1)
        {
            // Find the target epoch to align the population to.
            // TODO: Latest and earliest epoch will still consider the whole population,
            // even if an index list is given. This may throw off the percentage counter.
            const double mjd1950 = 2433282.5;
            double latestEpoch = population.getLatestEpoch();
            double earliestEpoch = population.getEarliestEpoch();
            if (latestEpoch < mjd1950)
            {
                Logger::out(0) << "Cannot align: Current epoch must be set for all objects." << std::endl;
                return INVALID_DATA;
            }
            if (toEpoch >= latestEpoch)
            {
                latestEpoch = toEpoch;
            }
            else if (toEpoch > 0.0)
            {
                Logger::out(0) << "Given epoch is too small. Using latest epoch from the population instead." << std::endl;
            }
            Logger::out(0) << "Aligning " << loopSize << " objects to epoch " << std::setprecision(15) << latestEpoch << ". This may take a while." << std::endl;

            int stepsRequired = int((latestEpoch - earliestEpoch)*86400.0 / dt) + 1;
            int stepsDone = 0;
            int percentDone = -1;

            ErrorCode error;
            int objectsAligned = 0;
            while (objectsAligned < loopSize)
            {
                objectsAligned = 0;
                IndexList trailingObjects(population.getHostPointer());
                for (int k=0; k<loopSize; k++)
                {
                    // Get next object index either from population or from index list
                    int i = (indices ? indices->getData(DEVICE_HOST)[k] : k);

                    // Find objects that are trailing behind the object with the latest current epoch.
                    const double currentEpoch = population.getEpoch()[i].current_epoch;
                    const double deltaSeconds = (latestEpoch - currentEpoch) * 86400.0;
                    if (deltaSeconds < 1)
                    {
                        // Current object is already aligned.
                        objectsAligned++;
                    }
                    else if (deltaSeconds >= dt)
                    {
                        // Current object is trailing by more than the given dt. Add to list.
                        trailingObjects.add(i);
                    }
                    else if (deltaSeconds < dt)
                    {
                        // Object is close to the target epoch. Propagate individually.
                        IndexList thisObject(population.getHostPointer());
                        thisObject.add(i);
                        Logger::out(0) << "Object " << i << " closing in. Propagating for " << deltaSeconds << " seconds." << std::endl;
                        error = propagate(population, 0.0, deltaSeconds, MODE_INDIVIDUAL_EPOCHS, &thisObject);
                    }
                }

                // Propagate all trailing objects.
                if (trailingObjects.getSize() > 0) error = propagate(population, 0.0, dt, MODE_INDIVIDUAL_EPOCHS, &trailingObjects);

                // Check for errors
                if (error == NOT_IMPLEMENTED)
                {
                    Logger::out(0) << "Cannot align: Propagator does not support the required functions." << std::endl;
                    return error;
                }
                else if (error != SUCCESS)
                {
                    Logger::out(0) << "Propagator returned an error during object alignment." << std::endl;
                    return error;
                }
                //Logger::out(0) << objectsAligned << " objects aligned." << std::endl;
                stepsDone++;
                int p = stepsDone * 100 / stepsRequired;
                if (p > percentDone)
                {
                    percentDone = p;
                    if (!quiet) Logger::out(0) << p << "% done." << std::endl;
                }
            }
        }
        else {
            Logger::out(0) << "Population too small - no alignment required." << std::endl;
        }
        //Logger::out(0) << "Alignment complete." << std::endl;
        return SUCCESS;
    }
}
