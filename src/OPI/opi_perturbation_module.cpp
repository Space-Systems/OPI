
#include "opi_perturbation_module.h"
namespace OPI
{
	class PerturbationModuleImpl
	{
		public:
	};

	PerturbationModule::PerturbationModule()
	{
	}

	PerturbationModule::~PerturbationModule()
	{
	}

    ErrorCode PerturbationModule::calculate(Population& population, Perturbations& delta, double julian_day, double dt, PropagationMode mode, IndexList* indices)
	{
        return runCalculation(population, delta, julian_day, dt, mode, indices);
	}

    ErrorCode PerturbationModule::runCalculation(Population& population, Perturbations& delta, double julian_day, double dt, PropagationMode mode, IndexList* indices)
	{
		return NOT_IMPLEMENTED;
	}

};
