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
#ifndef OPI_INTERNAL_PIMPL_HELPER_H
#define OPI_INTERNAL_PIMPL_HELPER_H

namespace OPI
{
	template<class T>
	//! Helper class for pimpl-idiom
	//!
	class Pimpl
	{
		public:
			//! allocate impl data on construction
			Pimpl() { data = new T; }
            template< class T2>
            Pimpl(T2& value) { data = new T(value); }

            Pimpl(T* value) { data = value; }

			~Pimpl() { delete data; }
			T* operator->() const { return data; }
			T* operator*() const { return data; }

		private:
			T* data;
	};
}

#endif
