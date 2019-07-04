#ifndef OPI_SYNCHRONIZED_DATA_H
#define OPI_SYNCHRONIZED_DATA_H
#include "../opi_host.h"
#include "opi_gpusupport.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
namespace OPI
{
	//! Template based inter-device synchronization helper class
	template< class DataType >
	class SynchronizedData
	{
		public:
            //! Initialize with a reference to the host object
            SynchronizedData(Host& owning_host);
            SynchronizedData(SynchronizedData<DataType>& source);
			~SynchronizedData();

			//! Reserves space to hold a specific amount of objects
			void reserve(int num_Objects);
			//! Resize all memory objects
            void resize(int num_Objects);

			//! Remove the object at index index;
            void remove(int index, int arraySize = 1);
			//! Retrieve the device-specific data pointer for the requested device
			DataType* getData(Device device, bool no_sync);
			//! Notify about updates in data structure of requested device
			void update(Device device);

			//! Returns the allocated number of objects
			int getReservedSize();
			//! Returns the number of used objects
			int getSize();

			//! Sorts the internal data
			void sort();

			//! Adds an object to the back of the host memory
			void add(const DataType& object);
            //! Adds a synchronized data object of the same type to the back of the host memory
            void add(SynchronizedData<DataType>& object);
			//! Sets an specific object
			void set(const DataType& object, int index);

			//! Checks if some data has been stored
			bool hasData();

			//! Removes duplicate data entries
			void removeDuplicates();
		private:
			//! Makes sure the data pointer on the specific device is allocated
			void ensure_allocation(Device device);
			//! Makes sure the data pointer on the specific device has up-to-date data
			void ensure_synchronization(Device device);
			//! Copies data from host to the specific device
			void sync_host_to_device(Device device);

			//! Clears all device data but keeps host data
			void clearDevices();

			//! the host memory
			std::vector<DataType> hostData;
			//! if the host needs an update
			bool hostNeedsUpdate;
			//! Device specific data container
			struct DeviceData
			{
					DeviceData(): ptr(0),needsUpdate(false)	{ }
					//! The pointer to the on-device memory data location
					DataType* ptr;
					//! If this device needs an update
					bool needsUpdate;
			};
			//! Reference to the host object
			Host& host;
			//! Map storing device specific data
			std::map<Device, DeviceData> deviceData;
			//! The device with the latest up-to-date data
			Device latestDevice;
			//! The number of objects this data object can currently hold
			int numObjects;
            int reservedSize;
	};

	template<class DataType>
    SynchronizedData<DataType>::SynchronizedData(Host& owning_host):
		host(owning_host)
	{
		// set latest device to -1
		latestDevice = DEVICE_NOT_SET;
		// the host does not need an update by default
		hostNeedsUpdate = false;
		numObjects = 0;
        reservedSize = 0;
	}

    // Copy constructor
    template<class DataType>
    SynchronizedData<DataType>::SynchronizedData(SynchronizedData<DataType>& source): host(source.host)
    {
        // download device data from source if necessary
        source.ensure_synchronization(DEVICE_HOST);

        latestDevice = DEVICE_NOT_SET;
        hostNeedsUpdate = false;

        numObjects = source.numObjects;
        reservedSize = source.reservedSize;
        hostData = source.hostData;

        // update host
        update(DEVICE_HOST);
    }

	template<class DataType>
	SynchronizedData<DataType>::~SynchronizedData()
	{
		// retrieve cuda support object
		GpuSupport* cuda = host.getGPUSupport();
		// check if the object is valid
		if(cuda) {
			// store currently selected device
			int oldDevice = cuda->getCurrentDevice();
			for(typename std::map<Device, DeviceData>::iterator itr = deviceData.begin(); itr != deviceData.end(); ++itr) {
				// check if the pointer is allocated (not 0)
				if(itr->second.ptr) {
					// select device
					cuda->selectDevice(itr->first - DEVICE_CUDA);
					// and free pointer
					cuda->free(itr->second.ptr);
				}
			}
			// select the old device again
			cuda->selectDevice(oldDevice);
		}
	}

	template<class DataType>
	void SynchronizedData<DataType>::add(const DataType &object)
	{
		ensure_synchronization(DEVICE_HOST);
		reserve(numObjects + 1);
		hostData.resize(numObjects + 1);
		hostData[numObjects] = object;
		resize(numObjects + 1);
		update(DEVICE_HOST);
	}

    template<class DataType>
    void SynchronizedData<DataType>::add(SynchronizedData<DataType> &objects)
    {
        objects.ensure_synchronization(DEVICE_HOST);
        int newDataSize = objects.getSize();
        ensure_synchronization(DEVICE_HOST);
        reserve(numObjects + newDataSize);
        hostData.insert(hostData.end(), objects.hostData.begin(), objects.hostData.end());
        resize(numObjects + newDataSize);
        update(DEVICE_HOST);
    }

	template<class DataType>
	void SynchronizedData<DataType>::set(const DataType &object, int index)
	{
		if((index >= 0)&&(index < numObjects))
		{
			ensure_synchronization(DEVICE_HOST);
			hostData[index] = object;
			update(DEVICE_HOST);
		}
	}

	template<class DataType>
	void SynchronizedData<DataType>::sort()
	{
		if(hasData())
		{
			ensure_synchronization(DEVICE_HOST);
			std::sort(hostData.begin(), hostData.end());

			update(DEVICE_HOST);
		}
	}

	template<class DataType>
	bool SynchronizedData<DataType>::hasData()
	{
		// check if there is any data stored
		bool hasDataStored = false;
		if(hostData.size() > 0)
			hasDataStored = true;
		for(typename std::map<Device, DeviceData>::iterator itr = deviceData.begin(); itr != deviceData.end(); ++itr) {
			// check if pointer is allocated
			if(itr->second.ptr) {
				hasDataStored = true;
			}
		}
		return hasDataStored;
	}


	template<class DataType>
	int SynchronizedData<DataType>::getReservedSize()
	{
		return reservedSize;
	}


	template<class DataType>
	int SynchronizedData<DataType>::getSize()
	{
		return numObjects;
	}

	template<class DataType>
	void SynchronizedData<DataType>::removeDuplicates()
	{
		ensure_synchronization(DEVICE_HOST);
		std::sort( hostData.begin(), hostData.end());
		hostData.erase( std::unique( hostData.begin(), hostData.end()), hostData.end() );
		numObjects = hostData.size();
		update(DEVICE_HOST);
	}

	template<class DataType>
	void SynchronizedData<DataType>::clearDevices()
	{
		// retrieve cuda support object
		GpuSupport* cuda = host.getGPUSupport();
		// check if the object is valid
		if(cuda) {
			// store current device
			int oldDevice = cuda->getCurrentDevice();
			// invalidate all device pointers
			for(typename std::map<Device, DeviceData>::iterator itr = deviceData.begin(); itr != deviceData.end(); ++itr) {
				// check if pointer is allocated
				if(itr->second.ptr) {
					// select device
					cuda->selectDevice(itr->first - DEVICE_CUDA);
					// free memory
					cuda->free(itr->second.ptr);
				}
				// reset to default values
				itr->second.ptr = 0;
				itr->second.needsUpdate = false;
			}
			// select the previously selected cuda device
			cuda->selectDevice(oldDevice);
		}
	}


	template<class DataType>
    void SynchronizedData<DataType>::remove(int index, int arraySize)
	{
		// check if data is available and the index range is valid
		if((hasData()) && (index >= 0) && (index < numObjects))
		{
			// synchronize data to host
			ensure_synchronization(DEVICE_HOST);
			// erase element from host vector
            hostData.erase(hostData.begin() + index, hostData.begin() + index + arraySize);
			// update where the latest information is located
			update(DEVICE_HOST);
			numObjects -= 1;
		}
	}

	template<class DataType>
	void SynchronizedData<DataType>::reserve(int num_Objects)
	{
		if(num_Objects > reservedSize)
		{
			if((hasData()))
			{
				// first synchronize data to host
				ensure_synchronization(DEVICE_HOST);
				// then delete all device data
				clearDevices();
				// resize the host vector
                hostData.reserve(num_Objects);
                // zero the new elements
                std::memset(hostData.data()+reservedSize, 0, (num_Objects-reservedSize)*sizeof(DataType));
                // now the host holds the latest information
				hostNeedsUpdate = false;
				latestDevice = DEVICE_HOST;
			}
			reservedSize = num_Objects;
			// store the number of allocated objects
		}
        else if (num_Objects < reservedSize)
		{
            for (int i=reservedSize-1; i>=num_Objects; i--)
            {
                // FIXME: Untested - is this enough?
                remove(i);
            }
		}
	}

	template<class DataType>
	void SynchronizedData<DataType>::resize(int num_Objects)
	{
		if(num_Objects > reservedSize)
		{
			reserve(num_Objects);
		}
		if(hostData.capacity() > 0)
		{
			hostData.resize(num_Objects);
		}
		numObjects = num_Objects;
    }

	template<class DataType>
	DataType* SynchronizedData<DataType>::getData(Device device, bool no_sync)
	{
		// no synchronization?
		if(no_sync) {
			// set update flag to false to suppress warnings
			if(device == DEVICE_HOST)
				hostNeedsUpdate = false;
			else
				deviceData[device].needsUpdate = false;
			// make sure the memory is allocated
			ensure_allocation(device);
		}
		else // we want a synchronization
			ensure_synchronization(device);
		if(device == DEVICE_HOST)
			return hostData.data();
		else
			return deviceData[device].ptr;
	}

	template<class DataType>
	void SynchronizedData<DataType>::ensure_allocation(Device device)
	{
		// host device?
		if(device == DEVICE_HOST) {
            size_t currentSize = hostData.size();
			if(hostData.capacity() != (size_t)reservedSize)
				hostData.reserve(reservedSize);
			hostData.resize(numObjects);
            if ((size_t)numObjects > currentSize)
                std::memset(hostData.data()+currentSize, 0, ((size_t)numObjects-currentSize)*sizeof(DataType));
		}
		else if ((device >= DEVICE_CUDA)&&(device <= DEVICE_CUDA_LAST)) {
			// retrieve cuda support object from host
			GpuSupport* cuda = host.getGPUSupport();
			// check if object is valid
			if(cuda) {
				// check if the requested device is not out of range
				if((device - DEVICE_CUDA) < cuda->getDeviceCount()) {
					// check if pointer already allocated
					if(!deviceData[device].ptr)
					{
						// if not change device
						int oldDevice = cuda->getCurrentDevice();
						cuda->selectDevice(device - DEVICE_CUDA);
						// allocate
                        cuda->allocate((void**)&(deviceData[device].ptr), sizeof(DataType) * reservedSize);
						// set needUpdate flag to true
						deviceData[device].needsUpdate = true;
						// select the old device
						cuda->selectDevice(oldDevice);
					}
				}
				else // unknown device
					host.sendError(INVALID_DEVICE);
			}
			else // no cuda support
				host.sendError(CUDA_REQUIRED);
		}
		else // unknown device
			host.sendError(INVALID_DEVICE);
	}

	template<class DataType>
	void SynchronizedData<DataType>::ensure_synchronization(Device device)
	{
		// first make sure the memory is allocated
		ensure_allocation(device);
		// only update if there is an update available
		if(latestDevice != -1) {
			// device is host device
			if(device == DEVICE_HOST) {
				// host device requires an update?
				if(hostNeedsUpdate) {
					// check if the device with the latest information is a cuda device
					if((latestDevice >= DEVICE_CUDA)&&(latestDevice <= DEVICE_CUDA_LAST)) {
						// retrieve cuda support object
						GpuSupport* cuda = host.getGPUSupport();
						// check if the cuda support is valid
						if(cuda) {
							// store current selected device
							int oldDevice = cuda->getCurrentDevice();
							// select new device
							cuda->selectDevice(latestDevice - DEVICE_CUDA);
							// copy data from device to host
							hostData.resize(numObjects);
                            cuda->copy(hostData.data(), deviceData[latestDevice].ptr, sizeof(DataType), numObjects, false);
							// set update flag to false, since we just updated the values
							hostNeedsUpdate = false;
							// select the old device
							cuda->selectDevice(oldDevice);
						}
						else // no cuda support
							host.sendError(CUDA_REQUIRED);
					}
					else // unknown device
						host.sendError(INVALID_DEVICE);
				}
			}
			else if ((device >= DEVICE_CUDA)&&(device <= DEVICE_CUDA_LAST)) {
				// check if pointer already allocated
				if(latestDevice == DEVICE_HOST) {
					sync_host_to_device(device);
				}
				else if((latestDevice >= DEVICE_CUDA)&&(latestDevice <= DEVICE_CUDA_LAST)) {
					// synchronize to host first
					ensure_synchronization(DEVICE_HOST);
					// now synchronize from our host to the requested device
					sync_host_to_device(device);
					// update the dirty flag
					deviceData[device].needsUpdate = false;
				}
				else // unknown device
					host.sendError(INVALID_DEVICE);
			}
			else // unknown device
				host.sendError(INVALID_DEVICE);
		}
	}

	template<class DataType>
	void SynchronizedData<DataType>::sync_host_to_device(Device device)
	{
		// retrieve cuda support
		GpuSupport* cuda = host.getGPUSupport();
		// check if support is valid
		if(cuda) {
			// store currently selected device
			int oldDevice = cuda->getCurrentDevice();
			// select the right device
			cuda->selectDevice(device - DEVICE_CUDA);
			// copy data from host to device
            cuda->copy(deviceData[device].ptr, hostData.data(), sizeof(DataType), numObjects, true);
			// select the old device again
			cuda->selectDevice(oldDevice);
		}
		else // cuda is required
			host.sendError(CUDA_REQUIRED);
	}

	template<class DataType>
	void SynchronizedData<DataType>::update(Device device)
	{
		latestDevice = device;
		// the host needs an update if the device is not the host itself
		hostNeedsUpdate = (device != DEVICE_HOST);
		for(typename std::map<Device, DeviceData>::iterator itr = deviceData.begin(); itr != deviceData.end(); ++itr)
		{
			// if the device has some memory allocated
			if(itr->second.ptr)
				// it may need an update
				itr->second.needsUpdate = (itr->first != device);
		}
	}
}

#endif
