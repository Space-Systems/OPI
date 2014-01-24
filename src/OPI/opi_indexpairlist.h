#ifndef OPI_INDEXPAIRLIST_H
#define OPI_INDEXPAIRLIST_H
#include "opi_common.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
namespace OPI
{
	class IndexPairListImpl;
	class Host;
	class IndexPair;

	//! \brief This class represents a list of object index pairs
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT IndexPairList
	{
		public:
			/// The host object must be valid
			IndexPairList(Host& host);
			~IndexPairList();

			//! Adds an indexpair to the list
			void add(const IndexPair& pair);
			void add(int object1, int object2);

			/// Reserve memory to hold space for numPairs index pairs
			void reserve(int numPairs);
			/// Update the data on a specific device
			void update(Device device, int numPairs);
			/// Returns the amount of stored pairs
			int getPairsUsed() const;
			/// Returns the amount of object pairs this list can store
			int getTotalSpace() const;

			void removeDuplicates();
			/// Returns a device-specific pointer to the data
			IndexPair* getData(Device device = DEVICE_HOST, bool no_sync = false) const;
		private:
			/// Private implementation details (pimpl-idiom)
			Pimpl<IndexPairListImpl> impl;
	};
}
#endif // OPI_INDEXPAIRLIST_H
