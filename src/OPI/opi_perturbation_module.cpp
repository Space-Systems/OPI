
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

	ErrorCode PerturbationModule::calculate(Population& data, Orbit* delta, float dt )
	{
		return runCalculation(data, delta, dt);
	}

	ErrorCode PerturbationModule::runCalculation(Population& data, Orbit* delta, float dt )
	{
		return NOT_IMPLEMENTED;
	}

	ErrorCode setTimeStep(double julian_day)
	{
		// overload if necessary
		return OPI::SUCCESS;
	}

};
