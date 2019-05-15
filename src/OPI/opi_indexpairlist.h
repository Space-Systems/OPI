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
#ifndef OPI_INDEXPAIRLIST_H
#define OPI_INDEXPAIRLIST_H
#include "opi_common.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
namespace OPI
{
	class IndexPairListImpl;
	class Host;
	struct IndexPair;

	//! \brief This class represents a list of object index pairs
	//! \ingroup CPP_API_GROUP
	class IndexPairList
	{
		public:
			/// The host object must be valid
			OPI_API_EXPORT IndexPairList(Host& host);
			OPI_API_EXPORT ~IndexPairList();

			//! Adds an indexpair to the list
			OPI_API_EXPORT void add(const IndexPair& pair);
			OPI_API_EXPORT void add(int object1, int object2);

			/// Reserve memory to hold space for numPairs index pairs
			OPI_API_EXPORT void reserve(int numPairs);
			/// Update the data on a specific device
			OPI_API_EXPORT void update(Device device, int numPairs);
			/// Returns the amount of stored pairs
			OPI_API_EXPORT int getPairsUsed() const;
			/// Returns the amount of object pairs this list can store
			OPI_API_EXPORT int getTotalSpace() const;

			OPI_API_EXPORT void removeDuplicates();
			/// Returns a device-specific pointer to the data
			IndexPair* getData(Device device = DEVICE_HOST, bool no_sync = false) const;
		private:
			/// Private implementation details (pimpl-idiom)
			Pimpl<IndexPairListImpl> impl;
	};
}
#endif // OPI_INDEXPAIRLIST_H
