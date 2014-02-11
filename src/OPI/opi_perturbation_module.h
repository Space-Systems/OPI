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
#ifndef OPI_PERTURBATION_MODULE_H
#define OPI_PERTURBATION_MODULE_H

#include "opi_common.h"
#include "opi_module.h"
namespace OPI
{
	class ObjectData;
	/*!
	 * \brief This class represents a pertubation module which can be used by a Propagator
	 *
	 * \ingroup CPP_API_GROUP
	 * \see Module, Host
	 */
	class PerturbationModule:
			public Module
	{
		public:
			//! Calculates the Pertubation for the passed dataset
			/**
			 * The calculated pertubation forces will be added to the values present in data_out
			 */
			ErrorCode calculate(ObjectData& data_in, ObjectData& data_out, float years, float seconds, float dt );
		protected:
			virtual ErrorCode runCalculation(ObjectData& data_in, ObjectData& data_out, float years, float seconds, float dt );
	};
}

#endif
