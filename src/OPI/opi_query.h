#ifndef OPI_QUERY_H
#define OPI_QUERY_H
#include "opi_common.h"
#include "opi_data.h"
#include "opi_error.h"
#include "opi_module.h"
#ifdef __cplusplus
#include <string>
namespace OPI
{
	class ObjectData;
	class IndexPairList;

	class DistanceQueryImpl;
	//! \brief This class represents a way to query the ObjectData about objects which
	//! are in a certain range to each other
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT DistanceQuery:
		public Module
	{
		public:
			DistanceQuery();
			virtual ~DistanceQuery();

			//! Rebuilds the internal structure
			ErrorCode rebuild(ObjectData& data);
			//! Make a query about objects which resides inside a cube of cube_size
			ErrorCode queryCubicPairs(ObjectData& data, IndexPairList& pairs, float cube_size);
			//! Tell the query object to visualize its internal structure
			void debugDraw();


		protected:
			//! Override this function to change the rebuild behaviour
			virtual ErrorCode runRebuild(ObjectData& data) = 0;
			//! Override this function to change the query behaviour
			virtual ErrorCode runCubicPairQuery(ObjectData& data, IndexPairList& pairs, float cube_size) = 0;
			//! Override this function to change the debug draw command
			virtual void runDebugDraw();

		private:
			DistanceQueryImpl* impl;
	};
}
#endif

#endif
