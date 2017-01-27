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
#ifndef OPI_DATA_H
#define OPI_DATA_H
#include "opi_common.h"
#include "opi_error.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
#include <string>
namespace OPI
{
	class ObjectRawData;
	class Host;
	class ObjectProperties;
	class Vector3;
	class IndexPair;
	class IndexList;

	/*! \brief This class contains all parameters required for processing orbital objects.
	 * \ingroup CPP_API_GROUP
	 *
	 * When a Population object is created with a specific size, arrays of the types Orbit,
	 * ObjectProperties and ObjectStatus are initialized with that size. These arrays are empty
	 * and must be filled with actual data by the host or the plugin.
	 */
	class OPI_API_EXPORT Population
	{
		public:
			//! Constructor
			Population(Host& host, int size = 0);
			//! Destructor
			~Population();

			//! Resizes the internal memory buffers
            void resize(int size, int byteArraySize = 1);
            //! Set the per-object size of the bytes buffer
            void resizeByteArray(int size);
			//! Returns the number of objects the internal buffers can hold
			int getSize() const;

			//! Removes an object
			void remove(int index);
			//! Removes a number of objects
			void remove(IndexList& list);

			//! Stores the Object Data to disk
			void write(const std::string& filename);
			//! Loads the Object Data from disk
			ErrorCode read(const std::string& filename);

			//! Notify about updates on the specified device
			ErrorCode update(int type, Device device = DEVICE_HOST);

			//! Retrieve the orbital parameters on the specified device
			Orbit* getOrbit(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the object properties on the specified device
			ObjectProperties* getObjectProperties(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the position in cartesian coordinates on the specified device
			Vector3* getCartesianPosition(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the velocity in cartesian coordinates on the specified device
			Vector3* getVelocity(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the acceleration in cartesian coordinates on the specified device
            Vector3* getAcceleration(Device device = DEVICE_HOST, bool no_sync = false) const;

            char* getBytes(Device device = DEVICE_HOST, bool no_sync = false) const;

            Population createSubPopulation(IndexList &list);
            //! Retrieve a deep copy of the population on the host
            Population copy();

            //! Perform a sanity check on the current population data and generate debug information
			std::string sanityCheck();

		private:
			//! Private implementation data
            Pimpl<ObjectRawData> data;
    };
}

#endif
