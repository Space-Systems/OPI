#include "opi_collisiondetection.h"
#include "opi_host.h"

namespace OPI
{


	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	class CollisionDetectionImpl
	{
		public:
	};

	//! \endcond

	CollisionDetection::CollisionDetection()
	{
		data = new CollisionDetectionImpl();
	}

	CollisionDetection::~CollisionDetection()
	{
		delete data;
	}

	ErrorCode CollisionDetection::detectPairs(ObjectData &data, DistanceQuery *query, IndexPairList &pairs_out, float time_passed)
	{
		ErrorCode status = NO_ERROR;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == NO_ERROR)
			status = runDetectPairs(data, query, pairs_out, time_passed);
		getHost()->sendError(status);
		return status;
	}

}
