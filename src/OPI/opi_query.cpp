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
		impl = new DistanceQueryImpl;
	}

	DistanceQuery::~DistanceQuery()
	{
		delete impl;
	}

	ErrorCode DistanceQuery::rebuild(ObjectData &data)
	{
		if(data.getSize() == 0)
			return NO_ERROR;
		ErrorCode status;
		// ensure this DistanceQuery is enabled
		status = enable();
		// an error occured?
		if(status == NO_ERROR)
			status = runRebuild(data);
		// forward propagation call
		getHost()->sendError(status);
		return status;
	}

	ErrorCode DistanceQuery::queryCubicPairs(ObjectData &data, IndexPairList& pairs, float cube_size)
	{
		if(data.getSize() == 0)
			return NO_ERROR;
		ErrorCode status;
		// ensure this DistanceQuery is enabled
		status = enable();
		// an error occured?
		if(status == NO_ERROR)
			status = runCubicPairQuery(data, pairs, cube_size);
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
