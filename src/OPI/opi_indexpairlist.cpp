#include "opi_indexpairlist.h"
#include "internal/opi_synchronized_data.h"
namespace OPI
{
	bool operator<(const IndexPair& pair1, const IndexPair& pair2)
	{
		if(pair1.object1 < pair2.object1)
			return true;
		if(pair1.object1 == pair2.object1)
		{
			return pair1.object2 < pair2.object2;
		}
		return false;
	}

	bool operator==(const IndexPair& pair1, const IndexPair& pair2)
	{
		if(pair1.object1 == pair2.object1)
			return pair1.object2 == pair2.object2;
		if(pair1.object1 == pair2.object2)
			return pair1.object2 == pair2.object1;
		return false;
	}
	bool operator!=(const IndexPair& pair1, const IndexPair& pair2)
	{
		return !(pair1 == pair2);
	}

	std::ostream& operator<<(std::ostream& out, const IndexPair& pair)
	{
		out << pair.object1 << ", " << pair.object2;
		return out;
	}

	/**
	 * @cond INTERNAL_DOCUMENTATION
	 */
	class IndexPairListImpl
	{
		public:
			IndexPairListImpl(Host& host): data(host) {}
			SynchronizedData<IndexPair> data;
	};
	/**
	 * @endcond
	 */

	IndexPairList::IndexPairList(Host &host):
		impl(host)
	{
	}

	IndexPairList::~IndexPairList()
	{
	}

	void IndexPairList::add(const IndexPair &pair)
	{
		impl->data.add(pair);
	}

	void IndexPairList::add(int object1, int object2)
	{
		IndexPair pair;
		pair.object1 = object1;
		pair.object2 = object2;
		add(pair);
	}

	void IndexPairList::reserve(int numPairs)
	{
		impl->data.reserve(numPairs);
	}

	int IndexPairList::getPairsUsed() const
	{
		return impl->data.getSize();
	}

	int IndexPairList::getTotalSpace() const
	{
		return impl->data.getReservedSize();
	}

	void IndexPairList::removeDuplicates()
	{
		for(int i = 0; i < getPairsUsed(); ++i)
		{
			IndexPair* pairs = getData();
			if(pairs[i].object1 > pairs[i].object2)
			{
				int tmp = pairs[i].object1;
				pairs[i].object1 = pairs[i].object2;
				pairs[i].object2 = tmp;
			}
		}
		impl->data.removeDuplicates();
	}

	IndexPair* IndexPairList::getData(Device device, bool no_sync) const
	{
		return impl->data.getData(device, no_sync);
	}

	void IndexPairList::update(Device device, int numPairs)
	{
		impl->data.update(device);
		impl->data.resize(numPairs);
	}
}
