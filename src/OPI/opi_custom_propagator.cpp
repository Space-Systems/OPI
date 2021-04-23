#include "opi_custom_propagator.h"

namespace OPI
{
	struct CustomPropagatorImpl
	{
		std::vector<PerturbationModule*> modules;
		PropagatorIntegrator* integrator;
	};

    CustomPropagator::CustomPropagator(const char* name)
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

    ErrorCode CustomPropagator::runPropagation(Population& population, JulianDay epoch, long dt, PropagationMode mode, IndexList* indices)
	{
		return SUCCESS;
	}

	int CustomPropagator::requiresCUDA()
	{
		return 0;
	}
}
