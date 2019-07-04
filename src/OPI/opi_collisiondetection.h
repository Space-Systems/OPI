#ifndef OPI_COLLISION_DETECTION_H
#define OPI_COLLISION_DETECTION_H
#include "opi_population.h"
#include "opi_error.h"
#include "opi_module.h"
#include <string>
namespace OPI
{
	class Population;
	class IndexList;
	class IndexPairList;
	class DistanceQuery;

	//! Contains the propagation implementation data
	class CollisionDetectionImpl;


	//! \brief This class implements a way to detect collision pairs in an object population
	//! \ingroup CPP_API_GROUP
	class CollisionDetection:
		public Module
	{
		public:
			CollisionDetection();
			virtual ~CollisionDetection();

			//! Detect colliding pairs and store them in pairs_out, use the specified query object
			ErrorCode detectPairs(Population& population, DistanceQuery* query, IndexPairList& pairs_out, float time_passed);
		private:
			//! Implementation of pair detection
			virtual ErrorCode runDetectPairs(Population& population, DistanceQuery* query, IndexPairList& pairs_out, float time_passed) = 0;
			Pimpl<CollisionDetectionImpl> data;
	};
}

#endif
