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
#include "opi_population.h"
#include "opi_host.h"
#include "opi_indexlist.h"
#include "internal/opi_cudasupport.h"
#include "internal/opi_synchronized_data.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
namespace OPI
{
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */

	// this holds all internal Population variables (pimpl)
	struct ObjectRawData
	{
			ObjectRawData(Host& _host):
				host(_host),
				data_orbit(host),
				data_properties(host),
				data_status(host),
				data_position(host),
				data_velocity(host),
				data_acceleration(host)
			{

			}

			Host& host;

			SynchronizedData<Orbit> data_orbit;
			SynchronizedData<ObjectProperties> data_properties;
			SynchronizedData<ObjectStatus> data_status;
			SynchronizedData<Vector3> data_position;
			SynchronizedData<Vector3> data_velocity;
			SynchronizedData<Vector3> data_acceleration;
			// data size
			int size;

	};
	/**
	 * \endcond
	 */

	Population::Population(Host& host, int size):
		data(host)
	{
		data->size = 0;
		resize(size);
	}

	Population::~Population()
	{
	}

	/**
	 * \detail
	 * The File will contain a 32-bit integer containing the number of Objects.
	 * Following are the blocks of data consisting of:
	 * A 32-bit integer declaring the type of the block, followed by another 32-bit integer defining the size of one entry
	 * followed by entry_size * number_of_objects bytes containing the actual data
	 *
	 * This will not work between machines with different endianness!
	 */
	void Population::write(const std::string& filename)
	{
		int temp;
		std::ofstream out(filename.c_str(), std::ofstream::binary);
		if(out.is_open())
		{
			out.write(reinterpret_cast<char*>(&data->size), sizeof(int));
			if(data->data_orbit.hasData())
			{
				temp = DATA_ORBIT;
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				temp = sizeof(Orbit);
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				out.write(reinterpret_cast<char*>(getOrbit()), sizeof(Orbit) * data->size);
			}
			if(data->data_properties.hasData())
			{
				temp = DATA_PROPERTIES;
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				temp = sizeof(ObjectProperties);
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				out.write(reinterpret_cast<char*>(getObjectProperties()), sizeof(ObjectProperties) * data->size);
			}
			if(data->data_status.hasData())
			{
				temp = DATA_STATUS;
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				temp = sizeof(ObjectStatus);
				out.write(reinterpret_cast<char*>(&temp), sizeof(int));
				out.write(reinterpret_cast<char*>(getObjectStatus()), sizeof(ObjectStatus) * data->size);
			}
		}
	}

	/**
	 * \detail
	 * See Population::write for more information
	 */
	ErrorCode Population::read(const std::string& filename)
	{
		int number_of_objects;

		std::ifstream in(filename.c_str(), std::ofstream::binary);
		if(in.is_open())
		{
			in.read(reinterpret_cast<char*>(&number_of_objects), sizeof(int));
			resize(number_of_objects);
			data->size = number_of_objects;
			while(in.good())
			{
				int type;
				int size;
				in.read(reinterpret_cast<char*>(&type), sizeof(int));
				if(!in.eof())
				{
					in.read(reinterpret_cast<char*>(&size), sizeof(int));
					switch(type)
					{
						case DATA_ORBIT:
							if(size == sizeof(Orbit))
							{
								Orbit* orbit = getOrbit(DEVICE_HOST, true);
								in.read(reinterpret_cast<char*>(orbit), sizeof(Orbit) * number_of_objects);
								data->data_orbit.update(DEVICE_HOST);
								break;
							}
						case DATA_PROPERTIES:
							if(size == sizeof(ObjectProperties))
							{
								ObjectProperties* prop = getObjectProperties(DEVICE_HOST, true);
								in.read(reinterpret_cast<char*>(prop), sizeof(ObjectProperties) * number_of_objects);
								data->data_properties.update(DEVICE_HOST);
								break;
							}
						case DATA_STATUS:
							if(size == sizeof(ObjectStatus))
							{
								ObjectStatus* status = getObjectStatus(DEVICE_HOST, true);
								in.read(reinterpret_cast<char*>(status), sizeof(ObjectStatus) * number_of_objects);
								data->data_status.update(DEVICE_HOST);
								break;
							}
						default:
							std::cout << "Found unknown block id " << type << std::endl;
							in.seekg(number_of_objects * size);
					}
				}
			}
		}
		return SUCCESS;
	}

	void Population::resize(int size)
	{
		if(data->size != size)
		{
			data->data_orbit.resize(size);
			data->data_status.resize(size);
			data->data_properties.resize(size);
			data->data_position.resize(size);
			data->data_velocity.resize(size);
			data->data_acceleration.resize(size);
			data->size = size;
		}
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	Orbit* Population::getOrbit(Device device, bool no_sync) const
	{
		return data->data_orbit.getData(device, no_sync);
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	ObjectProperties* Population::getObjectProperties(Device device, bool no_sync) const
	{
		return data->data_properties.getData(device, no_sync);
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	ObjectStatus* Population::getObjectStatus(Device device, bool no_sync) const
	{
		return data->data_status.getData(device, no_sync);
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	Vector3* Population::getCartesianPosition(Device device, bool no_sync) const
	{
		return data->data_position.getData(device, no_sync);
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	Vector3* Population::getVelocity(Device device, bool no_sync) const
	{
		return data->data_velocity.getData(device, no_sync);
	}

	/**
	 * @details
	 * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
	 * requested device.
	 */
	Vector3* Population::getAcceleration(Device device, bool no_sync) const
	{
		return data->data_acceleration.getData(device, no_sync);
	}

	void Population::remove(IndexList &list)
	{
		list.sort();
		int* listdata = list.getData(DEVICE_HOST);
		int offset = 0;
		for(int i = 0; i < list.getSize(); ++i)
		{
			remove(listdata[i] - offset);
			offset++;
		}
	}

	void Population::remove(int index)
	{
		data->data_acceleration.remove(index);
		data->data_orbit.remove(index);
		data->data_position.remove(index);
		data->data_properties.remove(index);
		data->data_status.remove(index);
		data->data_velocity.remove(index);
		data->size--;
	}

	ErrorCode Population::update(int type, Device device)
	{
		ErrorCode status = SUCCESS;
		switch(type)
		{
			case DATA_ORBIT:
				data->data_orbit.update(device);
				break;
			case DATA_PROPERTIES:
				data->data_properties.update(device);
				break;
			case DATA_STATUS:
				data->data_status.update(device);
				break;
			case DATA_VELOCITY:
				data->data_velocity.update(device);
				break;
			case DATA_CARTESIAN:
				data->data_position.update(device);
				break;
			case DATA_ACCELERATION:
				data->data_acceleration.update(device);
				break;
			default:
				status = INVALID_TYPE;
		}
		data->host.sendError(status);
		return status;
	}

	int Population::getSize() const
	{
		return data->size;
	}

}
