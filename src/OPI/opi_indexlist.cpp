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
