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
#include "opi_perturbations.h"
#include "opi_host.h"
#include "opi_indexlist.h"
#include "opi_gpusupport.h"
#include "internal/opi_synchronized_data.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string.h> //memcpy
namespace OPI
{
    /**
     * \cond INTERNAL_DOCUMENTATION
     */

    // this holds all internal Perturbations variables (pimpl)
    struct PerturbationRawData
    {
		PerturbationRawData(Host& _host) :
                host(_host),
                data_orbit(host),
                data_position(host),
                data_velocity(host),
                data_acceleration(host),
                data_vmatrix(host),
                data_bytes(host)
            {

            }

            Host& host;

            SynchronizedData<Orbit> data_orbit;
            SynchronizedData<Vector3> data_position;
            SynchronizedData<Vector3> data_velocity;
            SynchronizedData<Vector3> data_acceleration;
            SynchronizedData<VMatrix> data_vmatrix;
            SynchronizedData<char> data_bytes;

            // non-synchronized data
            std::string lastPropagatorName;

            // data size
            int size;
            int byteArraySize;
    };
    /**
     * \endcond
     */

    Perturbations::Perturbations(const Population& population): data(population.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        data->lastPropagatorName = population.getLastPropagatorName();
        resize(population.getSize());
    }

    Perturbations::Perturbations(const Perturbations& source) : data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        data->lastPropagatorName = source.getLastPropagatorName();
        int s = source.getSize();
        int b = source.getByteArraySize();
        resize(s);
        resizeByteArray(b);

        // TODO Use std::copy instead
        memcpy(getDeltaOrbit(), source.getDeltaOrbit(), s*sizeof(Orbit));
        memcpy(getDeltaPosition(), source.getDeltaPosition(), s*sizeof(Vector3));
        memcpy(getDeltaVelocity(), source.getDeltaVelocity(), s*sizeof(Vector3));
        memcpy(getDeltaAcceleration(), source.getDeltaAcceleration(), s*sizeof(Vector3));
        memcpy(getVMatrix(), source.getVMatrix(), s*sizeof(VMatrix));
        memcpy(getBytes(), source.getBytes(), b*s*sizeof(char));

        update(DATA_ORBIT);
        update(DATA_PROPERTIES);
        update(DATA_POSITION);
        update(DATA_VELOCITY);
        update(DATA_ACCELERATION);
        update(DATA_VMATRIX);
        update(DATA_BYTES);
    }

    Perturbations::Perturbations(const Perturbations& source, IndexList &list) : data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        data->lastPropagatorName = source.getLastPropagatorName();
        int s = list.getSize();
        int b = source.getByteArraySize();
        resize(s);
        resizeByteArray(b);
        int* listdata = list.getData(DEVICE_HOST);

        Orbit* orbits = source.getDeltaOrbit(DEVICE_HOST, false);
        Vector3* pos = source.getDeltaPosition(DEVICE_HOST, false);
        Vector3* vel = source.getDeltaVelocity(DEVICE_HOST, false);
        Vector3* acc = source.getDeltaAcceleration(DEVICE_HOST, false);
        VMatrix* vmx = source.getVMatrix(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getDeltaOrbit();
        Vector3* thisPos = getDeltaPosition();
        Vector3* thisVel = getDeltaVelocity();
        Vector3* thisAcc = getDeltaAcceleration();
        VMatrix* thisVmx = getVMatrix();
        char* thisBytes = getBytes();

        for(int i = 0; i < list.getSize(); ++i)
        {
            thisOrbit[i] = orbits[listdata[i]];
            thisPos[i] = pos[listdata[i]];
            thisVel[i] = vel[listdata[i]];
            thisAcc[i] = acc[listdata[i]];
            thisVmx[i] = vmx[listdata[i]];
            for (int j=0; j<b; j++)
            {
                thisBytes[i*b+j] = bytes[listdata[i]*b+j];
            }
        }

        update(DATA_ORBIT);
        update(DATA_PROPERTIES);
        update(DATA_POSITION);
        update(DATA_VELOCITY);
        update(DATA_ACCELERATION);
        update(DATA_VMATRIX);
        update(DATA_BYTES);
    }

    Perturbations::~Perturbations()
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
    void Perturbations::write(const std::string& filename)
    {
        int temp;
        int versionNumber = 1;
        int magic = 45323;
        int nameLength = data->lastPropagatorName.length();
        std::ofstream out(filename.c_str(), std::ofstream::binary);
        if(out.is_open())
        {
            out.write(reinterpret_cast<char*>(&magic), sizeof(int));
            out.write(reinterpret_cast<char*>(&versionNumber), sizeof(int));
            out.write(reinterpret_cast<char*>(&data->size), sizeof(int));
            out.write(reinterpret_cast<char*>(&nameLength), sizeof(int));
            out.write(reinterpret_cast<char*>(&data->lastPropagatorName), data->lastPropagatorName.length());
            if(data->data_orbit.hasData())
            {
                temp = DATA_ORBIT;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = sizeof(Orbit);
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getDeltaOrbit()), sizeof(Orbit) * data->size);
            }
            if(data->data_position.hasData())
            {
                temp = DATA_POSITION;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = sizeof(Vector3);
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getDeltaPosition()), sizeof(Vector3) * data->size);
            }
            if(data->data_velocity.hasData())
            {
                temp = DATA_VELOCITY;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = sizeof(Vector3);
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getDeltaVelocity()), sizeof(Vector3) * data->size);
            }
            if(data->data_acceleration.hasData())
            {
                temp = DATA_ACCELERATION;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = sizeof(Vector3);
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getDeltaAcceleration()), sizeof(Vector3) * data->size);
            }
            if(data->data_vmatrix.hasData())
            {
                temp = DATA_VMATRIX;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = sizeof(VMatrix);
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getVMatrix()), sizeof(VMatrix) * data->size);
            }
            if(data->data_bytes.hasData())
            {
                temp = DATA_BYTES;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                temp = data->byteArraySize;
                out.write(reinterpret_cast<char*>(&temp), sizeof(int));
                out.write(reinterpret_cast<char*>(getBytes()), data->byteArraySize * data->size);
            }
        }
    }

    /**
     * \detail
     * See Perturbations::write for more information
     */
    ErrorCode Perturbations::read(const std::string& filename)
    {
        int number_of_objects = 0;
        int magicNumber = 0;
        int versionNumber = 0;
        int propagatorNameLength = 0;
        char* propagatorName;

        std::ifstream in(filename.c_str(), std::ifstream::binary);
        if(in.is_open())
        {
            in.read(reinterpret_cast<char*>(&magicNumber), sizeof(int));
            if (magicNumber == 45323)
            {
                in.read(reinterpret_cast<char*>(&versionNumber), sizeof(int));
                if (versionNumber == 1)
                {
                    in.read(reinterpret_cast<char*>(&number_of_objects), sizeof(int));
                    resize(number_of_objects);
                    data->size = number_of_objects;
                    in.read(reinterpret_cast<char*>(&propagatorNameLength), sizeof(int));
                    in.read(reinterpret_cast<char*>(&propagatorName), propagatorNameLength*sizeof(char));
                    data->lastPropagatorName = std::string(propagatorName);
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
                                    Orbit* orbit = getDeltaOrbit(DEVICE_HOST, true);
                                    in.read(reinterpret_cast<char*>(orbit), sizeof(Orbit) * number_of_objects);
                                    data->data_orbit.update(DEVICE_HOST);
                                    break;
                                }
                            case DATA_POSITION:
                                if(size == sizeof(Vector3))
                                {
                                    Vector3* pos = getDeltaPosition(DEVICE_HOST, true);
                                    in.read(reinterpret_cast<char*>(pos), sizeof(Vector3) * number_of_objects);
                                    data->data_position.update(DEVICE_HOST);
                                    break;
                                }
                            case DATA_VELOCITY:
                                if(size == sizeof(Vector3))
                                {
                                    Vector3* vel = getDeltaVelocity(DEVICE_HOST, true);
                                    in.read(reinterpret_cast<char*>(vel), sizeof(Vector3) * number_of_objects);
                                    data->data_velocity.update(DEVICE_HOST);
                                    break;
                                }
                            case DATA_ACCELERATION:
                                if(size == sizeof(Vector3))
                                {
                                    Vector3* acc = getDeltaAcceleration(DEVICE_HOST, true);
                                    in.read(reinterpret_cast<char*>(acc), sizeof(Vector3) * number_of_objects);
                                    data->data_acceleration.update(DEVICE_HOST);
                                    break;
                                }
                            case DATA_VMATRIX:
                                if(size == sizeof(VMatrix))
                                {
                                    VMatrix* vmx = getVMatrix(DEVICE_HOST, true);
                                    in.read(reinterpret_cast<char*>(vmx), sizeof(VMatrix) * number_of_objects);
                                    data->data_vmatrix.update(DEVICE_HOST);
                                    break;
                                }
                            case DATA_BYTES:
                                if(size == size) //TODO
                                {
                                    resizeByteArray(size);
                                    char* bytes = getBytes(DEVICE_HOST, true);
                                    in.read(bytes, size * number_of_objects * sizeof(char));
                                    data->data_bytes.update(DEVICE_HOST);
                                    break;
                                }
                            default:
                                std::cout << "Found unknown block id " << type << std::endl;
                                in.seekg(number_of_objects * size);
                            }
                        }
                    }
                }
                else std::cout << "Unknown file version" << std::endl;
            }
            else std::cout << filename << " does not appear to be an OPI perturbations file." << std::endl;
        }
        return SUCCESS;
    }

    void Perturbations::resize(int size, int byteArraySize)
    {
        if (data->size != size)
        {
            data->data_orbit.resize(size);
            data->data_position.resize(size);
            data->data_velocity.resize(size);
            data->data_acceleration.resize(size);
            data->data_vmatrix.resize(size);
            data->data_bytes.resize(size*byteArraySize);
            //initialize new elements to zero
            if (size > data->size)
            {
                for (int i=data->size; i<size; i++)
                {
                    data->data_orbit.set(Orbit(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),i);
                    data->data_position.set(Vector3(0.0,0.0,0.0),i);
                    data->data_velocity.set(Vector3(0.0,0.0,0.0),i);
                    data->data_acceleration.set(Vector3(0.0,0.0,0.0),i);
                    data->data_vmatrix.set(VMatrix(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),i);
                }
                data->data_orbit.update(DEVICE_HOST);
                data->data_position.update(DEVICE_HOST);
                data->data_velocity.update(DEVICE_HOST);
                data->data_acceleration.update(DEVICE_HOST);
                data->data_vmatrix.update(DEVICE_HOST);
            }
            data->size = size;
            data->byteArraySize = byteArraySize;
        }
    }

    void Perturbations::resizeByteArray(int size)
    {
        data->data_bytes.resize(data->size * size);
        data->byteArraySize = size;
    }

    std::string Perturbations::getLastPropagatorName() const
    {
        return data->lastPropagatorName;
    }

    void Perturbations::setLastPropagatorName(std::string propagatorName)
    {
        data->lastPropagatorName = propagatorName;
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Orbit* Perturbations::getDeltaOrbit(Device device, bool no_sync) const
    {
        return data->data_orbit.getData(device, no_sync);
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Vector3* Perturbations::getDeltaPosition(Device device, bool no_sync) const
    {
        return data->data_position.getData(device, no_sync);
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Vector3* Perturbations::getDeltaVelocity(Device device, bool no_sync) const
    {
        return data->data_velocity.getData(device, no_sync);
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Vector3* Perturbations::getDeltaAcceleration(Device device, bool no_sync) const
    {
        return data->data_acceleration.getData(device, no_sync);
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    VMatrix* Perturbations::getVMatrix(Device device, bool no_sync) const
    {
        return data->data_vmatrix.getData(device, no_sync);
    }

    char* Perturbations::getBytes(Device device, bool no_sync) const
    {
        return data->data_bytes.getData(device, no_sync);
    }

    void Perturbations::remove(IndexList &list)
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

    void Perturbations::insert(Perturbations& source, IndexList& list)
    {
        int* listdata = list.getData(DEVICE_HOST);

        Orbit* orbits = source.getDeltaOrbit(DEVICE_HOST, false);
        Vector3* pos = source.getDeltaPosition(DEVICE_HOST, false);
        Vector3* vel = source.getDeltaVelocity(DEVICE_HOST, false);
        Vector3* acc = source.getDeltaAcceleration(DEVICE_HOST, false);
        VMatrix* vmx = source.getVMatrix(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getDeltaOrbit();
        Vector3* thisPos = getDeltaPosition();
        Vector3* thisVel = getDeltaVelocity();
        Vector3* thisAcc = getDeltaAcceleration();
        VMatrix* thisVmx = getVMatrix();
        char* thisBytes = getBytes();

        if (getByteArraySize() != source.getByteArraySize())
        {
            std::cout << "Warning: Cannot insert byte array into perturbations!" << std::endl;
        }

        if (list.getSize() >= source.getSize())
        {
            for(int i = 0; i < list.getSize(); ++i)
            {
                int l = listdata[i];
                if (l < getSize())
                {
                    thisOrbit[l] = orbits[i];
                    thisPos[l] = pos[i];
                    thisVel[l] = vel[i];
                    thisAcc[l] = acc[i];
                    thisVmx[l] = vmx[i];
                    if (getByteArraySize() == source.getByteArraySize())
                    {
                        int b = getByteArraySize();
                        for (int j=0; j<b; j++)
                        {
                            thisBytes[l*b+j] = bytes[i*b+j];
                        }
                    }
                }
                else {
                    std::cout << "Cannot insert - index out of range: " << l << std::endl;
                }
            }
        }
        else {
            std::cout << "Cannot insert - not enough elements in index list!" << std::endl;
        }

        update(DATA_ORBIT);
        update(DATA_POSITION);
        update(DATA_VELOCITY);
        update(DATA_ACCELERATION);
        update(DATA_VMATRIX);
        update(DATA_BYTES);
    }

    void Perturbations::remove(int index)
    {
        data->data_acceleration.remove(index);
        data->data_orbit.remove(index);
        data->data_position.remove(index);
        data->data_velocity.remove(index);
        data->data_vmatrix.remove(index);
        data->data_bytes.remove(index*data->byteArraySize, data->byteArraySize);
        data->size--;
    }

    ErrorCode Perturbations::update(int type, Device device)
    {
        ErrorCode status = SUCCESS;
        switch(type)
        {
            case DATA_ORBIT:
                data->data_orbit.update(device);
                break;
            case DATA_VELOCITY:
                data->data_velocity.update(device);
                break;
            case DATA_POSITION:
                data->data_position.update(device);
                break;
            case DATA_ACCELERATION:
                data->data_acceleration.update(device);
                break;
            case DATA_VMATRIX:
                data->data_vmatrix.update(device);
                break;
            case DATA_BYTES:
                data->data_bytes.update(device);
                break;
            default:
                status = INVALID_TYPE;
        }
        data->host.sendError(status);
        return status;
    }

    int Perturbations::getSize() const
    {
        return data->size;
    }

    int Perturbations::getByteArraySize() const
    {
        return data->byteArraySize;
    }

    Host& Perturbations::getHostPointer() const
    {
        return data->host;
    }
}
