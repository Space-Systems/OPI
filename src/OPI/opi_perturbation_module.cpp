
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

    ErrorCode PerturbationModule::calculate(Population& population, Perturbations& delta, double julian_day, double dt )
	{
        return runCalculation(population, delta, julian_day, dt);
	}

    ErrorCode PerturbationModule::runCalculation(Population& population, Perturbations& delta, double julian_day, double dt )
	{
		return NOT_IMPLEMENTED;
	}

};
