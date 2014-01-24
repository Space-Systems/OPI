#include "opi_custom_propagator.h"

namespace OPI
{
	CustomPropagator::CustomPropagator(const std::string &name)
	{
		setName(name);
		integrator = 0;
	}

	CustomPropagator::~CustomPropagator()
	{

	}

	void CustomPropagator::addModule(PropagatorModule *module)
	{
		modules.push_back(module);
	}

	void CustomPropagator::setIntegrator(PropagatorIntegrator *_integrator)
	{
		integrator = _integrator;
	}

	ErrorCode CustomPropagator::runPropagation(ObjectData& data, float years, float seconds, float dt )
	{
		return NO_ERROR;
	}
}
