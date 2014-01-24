#ifndef OPI_INDEXLIST_H
#define OPI_INDEXLIST_H
#include "opi_common.h"
#include "opi_datatypes.h"

#include "opi_pimpl_helper.h"
namespace OPI
{
	class IndexListImpl;
	class Host;
	class IndexPair;

	//! \brief This class represents a list of object indices
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT IndexList
	{
		public:
			//! The host object must be valid
			IndexList(Host& host);
			~IndexList();

			//! Adds an index to the list
			void add(int index);
			//! Sorts the list
			void sort();
			//! Reserve memory to hold space for numPairs indices
			void reserve(int numPairs);
			//! Update the data on a specific device
			void update(Device device, int numPairs);
			//! Returns the amount of stored indicies
			int getSize() const;
			//! Returns the amount of indices this list can store
			int getTotalSpace() const;
			//! Returns a device-specific pointer to the data
			int* getData(Device device, bool no_sync = false) const;

			//! Removes duplicate entries from this list
			void removeDuplicates();
		private:
			//! Private implementation details (pimpl-idiom)
			Pimpl<IndexListImpl> impl;
	};
}
#endif // OPI_INDEXPAIRLIST_H
