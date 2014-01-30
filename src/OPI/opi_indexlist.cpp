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
#include "opi_indexlist.h"
#include "internal/opi_synchronized_data.h"
namespace OPI
{
	/**
	 * @cond INTERNAL_DOCUMENTATION
	 */
	class IndexListImpl
	{
		public:
			IndexListImpl(Host& host): data(host) {}
			SynchronizedData<int> data;
	};
	/**
	 * @endcond
	 */

	IndexList::IndexList(Host &host):
		impl(host)
	{
	}

	IndexList::~IndexList()
	{
	}

	void IndexList::add(int index)
	{
		impl->data.add(index);
	}

	void IndexList::sort()
	{
		impl->data.sort();
	}

	void IndexList::reserve(int numPairs)
	{
		impl->data.reserve(numPairs);
	}

	int IndexList::getSize() const
	{
		return impl->data.getReservedSize();
	}

	int IndexList::getTotalSpace() const
	{
		return impl->data.getSize();
	}

	int* IndexList::getData(Device device, bool no_sync) const
	{
		return impl->data.getData(device, no_sync);
	}

	void IndexList::removeDuplicates()
	{
		impl->data.removeDuplicates();
	}

	void IndexList::update(Device device, int numPairs)
	{
		impl->data.resize(numPairs);
		impl->data.update(device);
	}
}
