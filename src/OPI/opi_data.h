#ifndef OPI_DATA_H
#define OPI_DATA_H
#include "opi_common.h"
#include "opi_error.h"
#include "opi_datatypes.h"
#include <string>
namespace OPI
{
	class ObjectRawData;
	class Host;
	class ObjectProperties;
	class ObjectStatus;
	class Vector3;
	class IndexPair;
	class IndexList;

	//! \brief This class contains all parameters required for processing orbital objects
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT ObjectData
	{
		public:
			//! Constructor
			ObjectData(Host& host, int size = 0);
			//! Destructor
			~ObjectData();

			//! Resizes the internal memory buffers
			void resize(int size);
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
			//! Retrieve the object status on the specified device
			ObjectStatus* getObjectStatus(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the position in cartesian coordinates on the specified device
			Vector3* getCartesianPosition(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the velocity in cartesian coordinates on the specified device
			Vector3* getVelocity(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the acceleration in cartesian coordinates on the specified device
			Vector3* getAcceleration(Device device = DEVICE_HOST, bool no_sync = false) const;

		private:
			//! Private implementation data
			ObjectRawData* data;
	};
}

#endif
