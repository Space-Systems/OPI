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

			~Pimpl() { delete data; }
			T* operator->() const { return data; }
			T* operator*() const { return data; }

		private:
			T* data;
	};
}

#endif
