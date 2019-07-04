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
	}

	CollisionDetection::~CollisionDetection()
	{
	}

	ErrorCode CollisionDetection::detectPairs(Population &population, DistanceQuery *query, IndexPairList &pairs_out, float time_passed)
	{
		ErrorCode status = SUCCESS;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == SUCCESS)
			status = runDetectPairs(population, query, pairs_out, time_passed);
		getHost()->sendError(status);
		return status;
	}

}
