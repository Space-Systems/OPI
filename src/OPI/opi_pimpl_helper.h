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

            // Make impl data copyable
            Pimpl(const Pimpl& source): data(new T(*source.data)) {}
            Pimpl& operator=(const Pimpl& source)
            {
                if (&source != this) {
                    delete data;
                    data = new T(*source.data);
                }
                return *this;
            }

            // Addition assignment operator for ObjectRawData and PerturbationRawData
            Pimpl& operator+=(const Pimpl& other) { *data += *other.data; return *this; }
            Pimpl operator+(const Pimpl& other) { return Pimpl(data+other.data); }

            ~Pimpl() { delete data; }
			T* operator->() const { return data; }
            T& operator*() const { return data; }

		private:
            T* data;
	};
}

#endif
