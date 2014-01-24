#include "opi_propagator.h"
#include "opi_host.h"

namespace OPI
{


	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	class PropagatorImpl
	{
		public:
	};

	//! \endcond

	Propagator::Propagator()
	{
		data = new PropagatorImpl();
	}

	Propagator::~Propagator()
	{
		delete data;
	}

	ErrorCode Propagator::propagate(ObjectData& objectdata, float years, float seconds, float dt)
	{
		ErrorCode status = NO_ERROR;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == NO_ERROR)
			status = runPropagation(objectdata, years, seconds, dt);
		getHost()->sendError(status);
		return status;
	}

	/**
	 * If the runPropagation method for index-based propagation is not overloaded (returning OPI_NOT_IMPLEMENTED)
	 * this function will perform a normal propagation instead.
	 */
	ErrorCode Propagator::propagate(ObjectData& objectdata, IndexList& indices, float years, float seconds, float dt)
	{
		ErrorCode status = NO_ERROR;
		status = runIndexedPropagation(objectdata, indices, years, seconds, dt);
		if(status == NOT_IMPLEMENTED)
			status = propagate(objectdata, years, seconds, dt);
		getHost()->sendError(status);
		return status;
	}

	bool Propagator::backwardPropagation()
	{
		return true;
	}

	ErrorCode Propagator::runIndexedPropagation(ObjectData& data, IndexList& indices, float years, float seconds, float dt)
	{
		return NOT_IMPLEMENTED;
	}

}
