#ifndef OPI_QUERY_H
#define OPI_QUERY_H
#include "opi_common.h"
#include "opi_population.h"
#include "opi_error.h"
#include "opi_module.h"
#include "opi_pimpl_helper.h"
#ifdef __cplusplus
#include <string>
namespace OPI
{
	class Population;
	class IndexPairList;

	class DistanceQueryImpl;
	//! \brief This class represents a way to query the Population about objects which
	//! are in a certain range to each other
	//! \ingroup CPP_API_GROUP
	class DistanceQuery:
		public Module
	{
		public:
			OPI_API_EXPORT DistanceQuery();
			OPI_API_EXPORT virtual ~DistanceQuery();

			//! Rebuilds the internal structure
			OPI_API_EXPORT ErrorCode rebuild(Population& population);
			//! Make a query about objects which resides inside a cube of cube_size
			OPI_API_EXPORT ErrorCode queryCubicPairs(Population& population, IndexPairList& pairs, float cube_size);
			//! Tell the query object to visualize its internal structure
			OPI_API_EXPORT void debugDraw();


		protected:
			//! Override this function to change the rebuild behaviour
			virtual ErrorCode runRebuild(Population& population) = 0;
			//! Override this function to change the query behaviour
			virtual ErrorCode runCubicPairQuery(Population& population, IndexPairList& pairs, float cube_size) = 0;
			//! Override this function to change the debug draw command
			virtual void runDebugDraw();

		private:
			Pimpl<DistanceQueryImpl> impl;
	};
}
#endif

#endif
