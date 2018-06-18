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
