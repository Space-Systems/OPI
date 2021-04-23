
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

    ErrorCode PerturbationModule::calculate(const Population& population, Perturbations& delta, JulianDay epoch, long dt, PropagationMode mode, IndexList* indices)
	{
        return runCalculation(population, delta, epoch, dt, mode, indices);
	}

    ErrorCode PerturbationModule::runCalculation(const Population& population, Perturbations& delta, JulianDay epoch, long dt, PropagationMode mode, IndexList* indices)
	{
		return NOT_IMPLEMENTED;
	}

};
