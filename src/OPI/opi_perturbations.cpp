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
#include "internal/miniz.h"
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
                data_partials(host),
                data_bytes(host)
            {

            }

            Host& host;

            SynchronizedData<Orbit> data_orbit;
            SynchronizedData<Vector3> data_position;
            SynchronizedData<Vector3> data_velocity;
            SynchronizedData<Vector3> data_acceleration;
            SynchronizedData<PartialsMatrix> data_partials;
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
        //data->lastPropagatorName = population.getLastPropagatorName();
        resize(population.getSize());
    }

    Perturbations::Perturbations(const Perturbations& source) : data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        //data->lastPropagatorName = source.getLastPropagatorName();
        int s = source.getSize();
        int b = source.getByteArraySize();
        resize(s,b);

        copy(source, 0, s, 0);
    }

    Perturbations::Perturbations(const Perturbations& source, IndexList &list) : data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        //data->lastPropagatorName = source.getLastPropagatorName();
        int s = list.getSize();
        int b = source.getByteArraySize();
        resize(s);
        resizeByteArray(b);
        int* listdata = list.getData(DEVICE_HOST);

        Orbit* orbits = source.getDeltaOrbit(DEVICE_HOST, false);
        Vector3* pos = source.getDeltaPosition(DEVICE_HOST, false);
        Vector3* vel = source.getDeltaVelocity(DEVICE_HOST, false);
        Vector3* acc = source.getDeltaAcceleration(DEVICE_HOST, false);
        PartialsMatrix* pmx = source.getPartialsMatrix(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getDeltaOrbit();
        Vector3* thisPos = getDeltaPosition();
        Vector3* thisVel = getDeltaVelocity();
        Vector3* thisAcc = getDeltaAcceleration();
        PartialsMatrix* thisPmx = getPartialsMatrix();
        char* thisBytes = getBytes();

        for(int i = 0; i < list.getSize(); ++i)
        {
            thisOrbit[i] = orbits[listdata[i]];
            thisPos[i] = pos[listdata[i]];
            thisVel[i] = vel[listdata[i]];
            thisAcc[i] = acc[listdata[i]];
            thisPmx[i] = pmx[listdata[i]];
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
        update(DATA_PARTIALS);
        update(DATA_BYTES);
    }

    Perturbations::~Perturbations()
    {
    }

    void Perturbations::append(const Perturbations& other)
    {
        const int oldSize = data->size;
        const int newSize = oldSize + other.getSize();

        // use the byte array size from this population
        // byte array data from appended population will only be copied
        // if it has the same size.
        resize(newSize, data->byteArraySize);

        copy(other, 0, other.getSize(), oldSize);

    }

    void Perturbations::copy(const Perturbations& source, int firstIndex, int length, int offset)
    {
        if ((offset + length) <= data->size)
        {
            bool copyBytes =(data->byteArraySize == source.getByteArraySize());
            if (!copyBytes) std::cout << "Warning: Copying perturbations without the byte array" << std::endl;

            // TODO Use std::copy instead
            memcpy(&getDeltaOrbit()[offset], &source.getDeltaOrbit(DEVICE_HOST, false)[firstIndex], length*sizeof(Orbit));
            memcpy(&getDeltaPosition()[offset], &source.getDeltaPosition(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getDeltaVelocity()[offset], &source.getDeltaVelocity(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getDeltaAcceleration()[offset], &source.getDeltaAcceleration(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getPartialsMatrix()[offset], &source.getPartialsMatrix(DEVICE_HOST, false)[firstIndex], length*sizeof(PartialsMatrix));
            if (copyBytes) memcpy(&getBytes()[offset], &source.getBytes(DEVICE_HOST, false)[firstIndex], data->byteArraySize*length*sizeof(char));
            else memset(&getBytes()[offset], 0, data->byteArraySize*length*sizeof(char));

            update(DATA_ORBIT);
            update(DATA_PROPERTIES);
            update(DATA_POSITION);
            update(DATA_VELOCITY);
            update(DATA_ACCELERATION);
            update(DATA_PARTIALS);
            if (copyBytes) update(DATA_BYTES);
        }
        else std::cout << "Cannot copy perturbation: Trying to copy " << length << " objects with offset " << offset << " but size is " << length << std::endl;
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
    void Perturbations::write(const char* filename)
    {
        int temp;
        int versionNumber = 1;
        int magic = 45323;
        int nameLength = data->lastPropagatorName.length();
        std::stringstream out;
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
        if(data->data_partials.hasData())
        {
            temp = DATA_PARTIALS;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(PartialsMatrix);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getPartialsMatrix()), sizeof(PartialsMatrix) * data->size);
        }
        if(data->data_bytes.hasData())
        {
            temp = DATA_BYTES;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = data->byteArraySize;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getBytes()), data->byteArraySize * data->size);
        }

        // Convert stringstream to byte array. Do not use .str() because it will terminate at \0.
        unsigned long uncompressedSize = out.tellp();
        char* bytes = new char[uncompressedSize];
        out.read(bytes, uncompressedSize);

        // Compress char array using miniz and write to file.
        unsigned long compressedSize = compressBound(uncompressedSize);
        unsigned char* compressedData = new unsigned char[compressedSize];
        int status = compress(compressedData, &compressedSize, (const unsigned char *)bytes, uncompressedSize);
        if (status == Z_OK)
        {
            std::ofstream outfile(filename, std::ofstream::binary);
            outfile.write((const char*)compressedData, compressedSize);
            // Append uncompressed data size
            outfile.write(reinterpret_cast<char*>(&uncompressedSize), sizeof(unsigned long));
            outfile.close();
        }
        else {
            std::cout << "Failed to compress perturbation data!" << std::endl;
        }
        delete[] compressedData;
        delete[] bytes;
    }

    /**
     * \detail
     * See Perturbations::write for more information
     */
    ErrorCode Perturbations::read(const char* filename)
    {
        std::ifstream infile(filename, std::ifstream::binary);
        if(infile.is_open())
        {
            infile.seekg(0, std::ios::end);
            // Last eight bytes are for the uncompressed data size
            size_t fileSize = (size_t)infile.tellg() - (size_t)sizeof(unsigned long);
            char* fileContents = new char[fileSize];
            infile.seekg(0, std::ios::beg);
            infile.read(fileContents, fileSize);
            unsigned long uncompressedSize = 0;
            infile.read(reinterpret_cast<char*>(&uncompressedSize), sizeof(unsigned long));
            infile.close();

            unsigned char* uncompressedData = new unsigned char[uncompressedSize];

            int status = uncompress(uncompressedData, &uncompressedSize, (const unsigned char*)fileContents, fileSize);
            delete[] fileContents;

            if (status == Z_OK)
            {
                std::stringstream in;
                in.write((char*)uncompressedData, (size_t)uncompressedSize);
                delete[] uncompressedData;

                int number_of_objects = 0;
                int magicNumber = 0;
                int versionNumber = 0;
                int propagatorNameLength = 0;
                char* propagatorName;


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
                                case DATA_PARTIALS:
                                    if(size == sizeof(PartialsMatrix))
                                    {
                                        PartialsMatrix* pmx = getPartialsMatrix(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(pmx), sizeof(PartialsMatrix) * number_of_objects);
                                        data->data_partials.update(DEVICE_HOST);
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
                else std::cout << std::string(filename) << " does not appear to be an OPI perturbations file." << std::endl;
            }
            else {
                std::cout << "Failed to decompress perturbation data! " << std::endl;
                delete[] uncompressedData;
            }
        }
        else std::cout << "Unable to open file " << filename << "!" << std::endl;
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
            data->data_partials.resize(size);
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
                    data->data_partials.set(PartialsMatrix(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),i); //FIXME: This sucks
                }
                data->data_orbit.update(DEVICE_HOST);
                data->data_position.update(DEVICE_HOST);
                data->data_velocity.update(DEVICE_HOST);
                data->data_acceleration.update(DEVICE_HOST);
                data->data_partials.update(DEVICE_HOST);
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

    const char* Perturbations::getLastPropagatorName() const
    {
        return (data->lastPropagatorName).c_str();
    }

    void Perturbations::setLastPropagatorName(const char* propagatorName)
    {
        data->lastPropagatorName = std::string(propagatorName);
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
    PartialsMatrix* Perturbations::getPartialsMatrix(Device device, bool no_sync) const
    {
        return data->data_partials.getData(device, no_sync);
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
        PartialsMatrix* pmx = source.getPartialsMatrix(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getDeltaOrbit();
        Vector3* thisPos = getDeltaPosition();
        Vector3* thisVel = getDeltaVelocity();
        Vector3* thisAcc = getDeltaAcceleration();
        PartialsMatrix* thisPmx = getPartialsMatrix();
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
                    thisPmx[l] = pmx[i];
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
        update(DATA_PARTIALS);
        update(DATA_BYTES);
    }

    void Perturbations::remove(int index)
    {
        data->data_acceleration.remove(index);
        data->data_orbit.remove(index);
        data->data_position.remove(index);
        data->data_velocity.remove(index);
        data->data_partials.remove(index);
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
            case DATA_PARTIALS:
                data->data_partials.update(device);
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
