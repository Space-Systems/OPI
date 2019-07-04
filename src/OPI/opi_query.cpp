#include "opi_query.h"
#include "opi_host.h"
namespace OPI
{
	//! \cond INTERNAL_DOCUMENTATION
	class DistanceQueryImpl
	{
		public:
	};
	//! \endcond

	DistanceQuery::DistanceQuery()
	{
	}

	DistanceQuery::~DistanceQuery()
	{
	}

	ErrorCode DistanceQuery::rebuild(Population &population)
	{
		if(population.getSize() == 0)
			return SUCCESS;
		ErrorCode status;
		// ensure this DistanceQuery is enabled
		status = enable();
		// an error occured?
		if(status == SUCCESS)
			status = runRebuild(population);
		// forward propagation call
		getHost()->sendError(status);
		return status;
	}

	ErrorCode DistanceQuery::queryCubicPairs(Population &population, IndexPairList& pairs, float cube_size)
	{
		if(population.getSize() == 0)
			return SUCCESS;
		ErrorCode status;
		// ensure this DistanceQuery is enabled
		status = enable();
		// an error occured?
		if(status == SUCCESS)
			status = runCubicPairQuery(population, pairs, cube_size);
		getHost()->sendError(status);
		// forward propagation call
		return status;
	}

	void DistanceQuery::debugDraw()
	{
		if(isEnabled())
			runDebugDraw();
	}

	void DistanceQuery::runDebugDraw()
	{

	}
}
