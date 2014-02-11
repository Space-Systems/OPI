/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
#ifndef OPI_QUERY_H
#define OPI_QUERY_H
#include "opi_common.h"
#include "opi_data.h"
#include "opi_error.h"
#include "opi_module.h"
#include "opi_pimpl_helper.h"
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
			Pimpl<DistanceQueryImpl> impl;
	};
}
#endif

#endif
