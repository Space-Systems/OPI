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
