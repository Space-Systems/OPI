
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

    ErrorCode PerturbationModule::calculate(Population& data, Perturbations& delta, double julian_day, double dt )
	{
        return runCalculation(data, delta, julian_day, dt);
	}

    ErrorCode PerturbationModule::runCalculation(Population& data, Perturbations& delta, double julian_day, double dt )
	{
		return NOT_IMPLEMENTED;
	}

};
