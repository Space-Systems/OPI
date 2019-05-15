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
#ifndef OPI_INDEXLIST_H
#define OPI_INDEXLIST_H
#include "opi_common.h"
#include "opi_datatypes.h"

#include "opi_pimpl_helper.h"
namespace OPI
{
	class IndexListImpl;
	class Host;
	struct IndexPair;

	//! \brief This class represents a list of object indices
	//! \ingroup CPP_API_GROUP
	class IndexList
	{
		public:
			//! The host object must be valid
			OPI_API_EXPORT IndexList(Host& host);
			OPI_API_EXPORT ~IndexList();

			//! Adds an index to the list
			OPI_API_EXPORT void add(int index);
			//! Sorts the list
			OPI_API_EXPORT void sort();
			//! Reserve memory to hold space for numPairs indices
			OPI_API_EXPORT void reserve(int numPairs);
			//! Update the data on a specific device
			OPI_API_EXPORT void update(Device device, int numPairs);
			//! Returns the amount of stored indicies
			OPI_API_EXPORT int getSize() const;
			//! Returns the amount of indices this list can store
			OPI_API_EXPORT int getTotalSpace() const;
			//! Returns a device-specific pointer to the data
			OPI_API_EXPORT int* getData(Device device, bool no_sync = false) const;

			//! Removes duplicate entries from this list
			void removeDuplicates();
		private:
			//! Private implementation details (pimpl-idiom)
			Pimpl<IndexListImpl> impl;
	};
}
#endif // OPI_INDEXPAIRLIST_H
