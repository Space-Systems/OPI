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
			ErrorCode detectPairs(Population& data, DistanceQuery* query, IndexPairList& pairs_out, float time_passed);
		private:
			//! Implementation of pair detection
			virtual ErrorCode runDetectPairs(Population& data, DistanceQuery* query, IndexPairList& pairs_out, float time_passed) = 0;
			Pimpl<CollisionDetectionImpl> data;
	};
}

#endif
