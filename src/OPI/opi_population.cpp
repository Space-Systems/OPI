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
#include "opi_gpusupport.h"
#include "opi_indexlist.h"
#include "internal/opi_synchronized_data.h"
#include "internal/miniz.h"
#include "internal/json.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string.h> //memcpy
#define _USE_MATH_DEFINES
#include <math.h>

using json = nlohmann::json;

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
				data_position(host),
				data_velocity(host),
                data_acceleration(host),
                data_epoch(host),
                data_covariance(host),
                data_bytes(host)
			{
                size = 0;
                byteArraySize = 1;
                lastPropagatorName = "None";
                description = "";
                frame = REF_UNSPECIFIED;
			}

            ObjectRawData(const ObjectRawData& source):
                host(source.host),
                data_orbit(source.data_orbit),
                data_properties(source.data_properties),
                data_position(source.data_position),
                data_velocity(source.data_velocity),
                data_acceleration(source.data_acceleration),
                data_epoch(source.data_epoch),
                data_covariance(source.data_covariance),
                data_bytes(source.data_bytes)
            {
                object_names = source.object_names;
                lastPropagatorName = source.lastPropagatorName;
                description = source.description;
                frame = source.frame;

                size = source.size;
                byteArraySize = source.byteArraySize;
            }

            ~ObjectRawData() {}

            ObjectRawData* clone() { return new ObjectRawData(*this); }

			Host& host;

			SynchronizedData<Orbit> data_orbit;
			SynchronizedData<ObjectProperties> data_properties;
			SynchronizedData<Vector3> data_position;
			SynchronizedData<Vector3> data_velocity;
            SynchronizedData<Vector3> data_acceleration;
            SynchronizedData<Epoch> data_epoch;
            SynchronizedData<Covariance> data_covariance;
            SynchronizedData<char> data_bytes;

            // non-synchronized data
            std::vector<std::string> object_names;
            std::string lastPropagatorName;
            std::string description;
            ReferenceFrame frame;

			// data size
			int size;
            int byteArraySize;
	};
	/**
	 * \endcond
	 */

    Population::Population(Host& host, int size): data(host)
	{
		resize(size);        
	}

    Population::Population(const Population& source): data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        data->lastPropagatorName = source.getLastPropagatorName();
        data->description = source.getDescription();
        data->frame = source.getReferenceFrame();
        int s = source.getSize();
        int b = source.getByteArraySize();
        resize(s,b);

        copy(source, 0, s, 0);
    }

    Population::Population(const Population& source, IndexList &list) : data(source.getHostPointer())
    {
        data->size = 0;
        data->byteArraySize = 1;
        data->lastPropagatorName = source.getLastPropagatorName();
        data->description = source.getDescription();
        data->frame = source.getReferenceFrame();
        int s = list.getSize();
        int b = source.getByteArraySize();
        resize(s);
        resizeByteArray(b);
        int* listdata = list.getData(DEVICE_HOST);

        Orbit* orbits = source.getOrbit(DEVICE_HOST, false);
        ObjectProperties* props = source.getObjectProperties(DEVICE_HOST, false);
        Vector3* pos = source.getPosition(DEVICE_HOST, false);
        Vector3* vel = source.getVelocity(DEVICE_HOST, false);
        Vector3* acc = source.getAcceleration(DEVICE_HOST, false);
        Epoch* ep = source.getEpoch(DEVICE_HOST, false);
        Covariance* cov = source.getCovariance(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getOrbit();
        ObjectProperties* thisProps = getObjectProperties();
        Vector3* thisPos = getPosition();
        Vector3* thisVel = getVelocity();
        Vector3* thisAcc = getAcceleration();
        Epoch* thisEp = getEpoch();
        Covariance* thisCov = getCovariance();
        char* thisBytes = getBytes();

        for(int i = 0; i < list.getSize(); ++i)
        {
            thisOrbit[i] = orbits[listdata[i]];
            thisProps[i] = props[listdata[i]];
            thisPos[i] = pos[listdata[i]];
            thisVel[i] = vel[listdata[i]];
            thisAcc[i] = acc[listdata[i]];
            thisEp[i] = ep[listdata[i]];
            thisCov[i] = cov[listdata[i]];
            data->object_names[i] = source.getObjectName(i);
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
        update(DATA_EPOCH);
        update(DATA_COVARIANCE);
        update(DATA_BYTES);
    }

	Population::~Population()
	{
    }

    void Population::append(const Population& other)
    {
        const int oldSize = data->size;
        const int newSize = oldSize + other.getSize();

        // use the byte array size from this population
        // byte array data from appended population will only be copied
        // if it has the same size.
        resize(newSize, data->byteArraySize);

        copy(other, 0, other.getSize(), oldSize);
    }

    void Population::copy(const Population& source, int firstIndex, int length, int offset)
    {
        if ((offset + length) <= data->size)
        {
            bool copyBytes = (data->byteArraySize == source.getByteArraySize());
            //if (!copyBytes) std::cout << "Warning: Copying population without the byte array" << std::endl;

            // TODO Use std::copy instead
            memcpy(&getOrbit()[offset], &source.getOrbit(DEVICE_HOST, false)[firstIndex], length*sizeof(Orbit));
            memcpy(&getObjectProperties()[offset], &source.getObjectProperties(DEVICE_HOST, false)[firstIndex], length*sizeof(ObjectProperties));
            memcpy(&getPosition()[offset], &source.getPosition(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getVelocity()[offset], &source.getVelocity(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getAcceleration()[offset], &source.getAcceleration(DEVICE_HOST, false)[firstIndex], length*sizeof(Vector3));
            memcpy(&getEpoch()[offset], &source.getEpoch(DEVICE_HOST, false)[firstIndex], length*sizeof(Epoch));
            memcpy(&getCovariance()[offset], &source.getCovariance(DEVICE_HOST, false)[firstIndex], length*sizeof(Covariance));
            for (int i=0; i<length; i++) data->object_names[offset+i] = source.getObjectName(firstIndex+i);
            if (copyBytes) memcpy(&getBytes()[offset], &source.getBytes(DEVICE_HOST, false)[firstIndex], data->byteArraySize*length*sizeof(char));
            else memset(&getBytes()[offset], 0, data->byteArraySize*length*sizeof(char));

            update(DATA_ORBIT);
            update(DATA_PROPERTIES);
            update(DATA_POSITION);
            update(DATA_VELOCITY);
            update(DATA_ACCELERATION);
            update(DATA_EPOCH);
            update(DATA_COVARIANCE);
            if (copyBytes) update(DATA_BYTES);
        }
        else std::cout << "Cannot copy population: Trying to copy " << length << " objects with offset " << offset << " but size is " << length << std::endl;
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
    void Population::write(const char* filename)
    {
		int temp;
        int versionNumber = OPI_DATA_REVISION_NUMBER;
        int magic = 47627;
        int nameLength = data->lastPropagatorName.length();        
        int descLength = data->description.length();
        std::string frame = std::string(referenceFrameToString(data->frame));
        int rfLength = frame.length();
        std::stringstream out;
        out.write(reinterpret_cast<char*>(&magic), sizeof(int));
        out.write(reinterpret_cast<char*>(&versionNumber), sizeof(int));
        out.write(reinterpret_cast<char*>(&data->size), sizeof(int));
        out.write(reinterpret_cast<char*>(&nameLength), sizeof(int));
        out.write(reinterpret_cast<const char*>(data->lastPropagatorName.c_str()), nameLength);
        out.write(reinterpret_cast<char*>(&descLength), sizeof(int));
        out.write(reinterpret_cast<const char*>(data->description.c_str()), descLength);
        out.write(reinterpret_cast<char*>(&rfLength), sizeof(int));
        out.write(reinterpret_cast<const char*>(frame.c_str()), rfLength);
        for (int i=0; i<data->size; i++)
        {
            int objectNameLength = data->object_names[i].length();
            out.write(reinterpret_cast<char*>(&objectNameLength), sizeof(int));
            if (objectNameLength > 0)
            {
                out.write(reinterpret_cast<const char*>(data->object_names[i].c_str()), objectNameLength);
            }
        }
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
        if(data->data_position.hasData())
        {
            temp = DATA_POSITION;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(Vector3);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getPosition()), sizeof(Vector3) * data->size);
        }
        if(data->data_velocity.hasData())
        {
            temp = DATA_VELOCITY;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(Vector3);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getVelocity()), sizeof(Vector3) * data->size);
        }
        if(data->data_acceleration.hasData())
        {
            temp = DATA_ACCELERATION;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(Vector3);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getAcceleration()), sizeof(Vector3) * data->size);
        }
        if(data->data_epoch.hasData())
        {
            temp = DATA_EPOCH;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(Epoch);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getEpoch()), sizeof(Epoch) * data->size);
        }
        if(data->data_covariance.hasData())
        {
            temp = DATA_COVARIANCE;
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            temp = sizeof(Covariance);
            out.write(reinterpret_cast<char*>(&temp), sizeof(int));
            out.write(reinterpret_cast<char*>(getCovariance()), sizeof(Covariance) * data->size);
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
        unsigned long long uncompressedSize = out.tellp();
        char* bytes = new char[uncompressedSize];
        out.read(bytes, uncompressedSize);

        // Compress char array using miniz and write to file.
        unsigned long compressedSize = compressBound(uncompressedSize);
        unsigned char* compressedData = new unsigned char[compressedSize];
        int status = compress(compressedData, (mz_ulong*)&compressedSize, (const unsigned char *)bytes, (mz_ulong)uncompressedSize);
        if (status == Z_OK)
        {
            std::ofstream outfile(filename, std::ofstream::binary);
            outfile.write((const char*)compressedData, compressedSize);
            // Append uncompressed data size
            outfile.write(reinterpret_cast<char*>(&uncompressedSize), sizeof(unsigned long long));
            outfile.close();
        }
        else {
            std::cout << "Failed to compress population data!" << std::endl;
        }
        delete[] compressedData;
        delete[] bytes;
    }

	/**
	 * \detail
	 * See Population::write for more information
	 */
    ErrorCode Population::read(const char* filename)
	{
        std::ifstream infile(filename, std::ifstream::binary);
        if (infile.is_open())
        {
            infile.seekg(0, std::ios::end);
            // Last eight bytes are for the uncompressed data size
            // Use unsigned long long because it has the same size on all platforms
            // Uncompressed file size is still limited to 4GB by mz_ulong
            size_t fileSize = (size_t)infile.tellg() - (size_t)sizeof(unsigned long long);
            char* fileContents = new char[fileSize];
            infile.seekg(0, std::ios::beg);
            infile.read(fileContents, fileSize);
            unsigned long long uncompressedSize = 0;
            infile.read(reinterpret_cast<char*>(&uncompressedSize), sizeof(unsigned long long));
            infile.close();

            unsigned char* uncompressedData = new unsigned char[uncompressedSize];

            int status = uncompress(uncompressedData, (mz_ulong*)&uncompressedSize, (const unsigned char*)fileContents, fileSize);
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
                int descLength = 0;
                data->frame = REF_UNSPECIFIED;

                in.read(reinterpret_cast<char*>(&magicNumber), sizeof(int));
                if (magicNumber == 47627)
                {
                    in.read(reinterpret_cast<char*>(&versionNumber), sizeof(int));
                    if (versionNumber >= 1)
                    {
                        in.read(reinterpret_cast<char*>(&number_of_objects), sizeof(int));
                        resize(number_of_objects);
                        data->size = number_of_objects;
                        in.read(reinterpret_cast<char*>(&propagatorNameLength), sizeof(int));
                        char* propagatorName = new char[propagatorNameLength];
                        in.read(propagatorName, propagatorNameLength);
                        if (versionNumber >= 2)
                        {
                            data->lastPropagatorName = std::string(propagatorName, propagatorNameLength);
                            in.read(reinterpret_cast<char*>(&descLength), sizeof(int));
                            char* description = new char[descLength];
                            in.read(description, descLength);
                            data->description = std::string(description, descLength);
                            delete[] description;
                            if (versionNumber >= 3)
                            {
                                int rfLength = 0;
                                in.read(reinterpret_cast<char*>(&rfLength), sizeof(int));
                                char* rfName = new char[rfLength];
                                in.read(rfName, rfLength);
                                data->frame = referenceFrameFromString(rfName);
                                delete[] rfName;
                            }
                        }
                        for (int i=0; i<data->size; i++)
                        {
                            int objectNameLength = 0;
                            in.read(reinterpret_cast<char*>(&objectNameLength),sizeof(int));
                            if (objectNameLength > 0)
                            {
                                char* objectName = new char[objectNameLength];
                                in.read(objectName, objectNameLength);
                                data->object_names[i] = std::string(objectName, objectNameLength);
                                delete[] objectName;
                            }
                        }
                        delete[] propagatorName;
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
                                case DATA_POSITION:
                                    if(size == sizeof(Vector3))
                                    {
                                        Vector3* pos = getPosition(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(pos), sizeof(Vector3) * number_of_objects);
                                        data->data_position.update(DEVICE_HOST);
                                        break;
                                    }
                                case DATA_VELOCITY:
                                    if(size == sizeof(Vector3))
                                    {
                                        Vector3* vel = getVelocity(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(vel), sizeof(Vector3) * number_of_objects);
                                        data->data_velocity.update(DEVICE_HOST);
                                        break;
                                    }
                                case DATA_ACCELERATION:
                                    if(size == sizeof(Vector3))
                                    {
                                        Vector3* acc = getAcceleration(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(acc), sizeof(Vector3) * number_of_objects);
                                        data->data_acceleration.update(DEVICE_HOST);
                                        break;
                                    }
                                case DATA_EPOCH:
                                    if(size == sizeof(Epoch))
                                    {
                                        Epoch* ep = getEpoch(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(ep), sizeof(Epoch) * number_of_objects);
                                        data->data_epoch.update(DEVICE_HOST);
                                        break;
                                    }
                                case DATA_COVARIANCE:
                                    if(size == sizeof(Covariance))
                                    {
                                        Covariance* cov = getCovariance(DEVICE_HOST, true);
                                        in.read(reinterpret_cast<char*>(cov), sizeof(Covariance) * number_of_objects);
                                        data->data_covariance.update(DEVICE_HOST);
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
                else std::cout << filename << " does not appear to be an OPI population file." << std::endl;
            }
            else {
                std::cout << "Failed to decompress population data! " << std::endl;
                delete[] uncompressedData;
            }
        }
        else std::cout << "Unable to open file " << filename << "!" << std::endl;
		return SUCCESS;
	}

    void Population::writeJSON(const char* filename)
    {
        json objects;
        for (int i=0; i<getSize(); i++)
        {
            json o;
            Vector3 p = getPosition()[i];
            Vector3 v = getVelocity()[i];
            Vector3 a = getAcceleration()[i];
            Orbit orb = getOrbit()[i];
            Epoch e = getEpoch()[i];
            ObjectProperties pr = getObjectProperties()[i];
            Covariance c = getCovariance()[i];
            if (std::string(getObjectName(i)) != "")
                o["name"] = getObjectName(i);
            if (!isZero(p))
                o["position"] = {{"x",p.x}, {"y",p.y}, {"z",p.z}};
            if (!isZero(v))
                o["velocity"] = {{"x",v.x}, {"y",v.y}, {"z",v.z}};
            if (!isZero(a))
                o["acceleration"] = {{"x",a.x}, {"y",a.y}, {"z",a.z}};
            if (!isZero(orb))
                o["orbit"] = {{"sma",orb.semi_major_axis}, {"ecc",orb.eccentricity}, {"inc",orb.inclination}, {"raan",orb.raan}, {"aop",orb.arg_of_perigee}, {"ma",orb.mean_anomaly}};
            if (!isZero(e))
                o["epoch"] = {{"bol",e.beginning_of_life}, {"eol",e.end_of_life}, {"current",e.current_epoch}};
            if (!isZero(pr))
                o["properties"] = {{"id",pr.id},{"mass",pr.mass},{"dia",pr.diameter},{"a2m",pr.area_to_mass},{"cd",pr.drag_coefficient},{"cr",pr.reflectivity}};
            if (!isZero(c))
                o["covariance"] = {c.k1_k1, c.k2_k1, c.k2_k2, c.k3_k1, c.k3_k2, c.k3_k3, c.k4_k1, c.k4_k2, c.k4_k3, c.k4_k4, c.k5_k1, c.k5_k2, c.k5_k3, c.k5_k4, c.k5_k5,
                        c.k6_k1, c.k6_k2, c.k6_k3, c.k6_k4, c.k6_k5, c.k6_k6, c.d1_k1, c.d1_k2, c.d1_k3, c.d1_k4, c.d1_k5, c.d1_k6, c.d1_d1,
                        c.d2_k1, c.d2_k2, c.d2_k3, c.d2_k4, c.d2_k5, c.d2_k6, c.d2_d1, c.d2_d2};
            objects.push_back(o);
        }
        json jp;
        jp["data_revision"] = OPI_DATA_REVISION_NUMBER;
        jp["description"] = getDescription();
        jp["epoch_earliest"] = getEarliestEpoch();
        jp["epoch_latest"] = getLatestEpoch();
        jp["objects"] = objects;
        std::string rf = referenceFrameToString(data->frame);
        if (rf != "") jp["frame"] = rf;
        std::ofstream outfile(filename);
        std::string jsonString = jp.dump(2);
        outfile.write(jsonString.c_str(), jsonString.size());
        outfile.close();
    }

    void Population::resize(int size, int byteArraySize)
	{
		if(data->size != size)
		{
			data->data_orbit.resize(size);
			data->data_properties.resize(size);
			data->data_position.resize(size);
			data->data_velocity.resize(size);
            data->data_acceleration.resize(size);
            data->data_epoch.resize(size);
            data->data_covariance.resize(size);
            data->data_bytes.resize(size*byteArraySize);
            data->object_names.resize(size);
			data->size = size;
            data->byteArraySize = byteArraySize;
		}
	}

    void Population::resizeByteArray(int size)
    {
        data->data_bytes.resize(data->size * size);
        data->byteArraySize = size;
    }

    const char* Population::getLastPropagatorName() const
    {
        return data->lastPropagatorName.c_str();
    }

    const char* Population::getDescription() const
    {
        return data->description.c_str();
    }

    ReferenceFrame Population::getReferenceFrame() const
    {
        return data->frame;
    }

    void Population::setLastPropagatorName(const char* propagatorName)
    {
        data->lastPropagatorName = std::string(propagatorName);
    }

    void Population::setDescription(const char* description)
    {
        data->description = std::string(description);
    }

    void Population::setReferenceFrame(const ReferenceFrame referenceFrame)
    {
        data->frame = referenceFrame;
    }

    const char* Population::getObjectName(int index) const
    {
        if (index < data->size)
            return data->object_names[index].c_str();
        else return "";
    }

    void Population::setObjectName(int index, const char* name)
    {
        if (index < data->size)
        {
            data->object_names[index] = std::string(name);
        }
        else std::cout << "Cannot set object name: Index (" << index << ") out of range!" << std::endl;
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
    Vector3* Population::getPosition(Device device, bool no_sync) const
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

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Epoch* Population::getEpoch(Device device, bool no_sync) const
    {
        return data->data_epoch.getData(device, no_sync);
    }

    /**
     * @details
     * If no_sync is set to false, a synchronization is performed to ensure the latest up-to-date data on the
     * requested device.
     */
    Covariance* Population::getCovariance(Device device, bool no_sync) const
    {
        return data->data_covariance.getData(device, no_sync);
    }

    char* Population::getBytes(Device device, bool no_sync) const
    {
        return data->data_bytes.getData(device, no_sync);
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

    void Population::insert(Population& source, IndexList& list)
    {
        int* listdata = list.getData(DEVICE_HOST);

        Orbit* orbits = source.getOrbit(DEVICE_HOST, false);
        ObjectProperties* props = source.getObjectProperties(DEVICE_HOST, false);
        Vector3* pos = source.getPosition(DEVICE_HOST, false);
        Vector3* vel = source.getVelocity(DEVICE_HOST, false);
        Vector3* acc = source.getAcceleration(DEVICE_HOST, false);
        Epoch* ep = source.getEpoch(DEVICE_HOST, false);
        Covariance* cov = source.getCovariance(DEVICE_HOST, false);
        char* bytes = source.getBytes(DEVICE_HOST, false);

        Orbit* thisOrbit = getOrbit();
        ObjectProperties* thisProps = getObjectProperties();
        Vector3* thisPos = getPosition();
        Vector3* thisVel = getVelocity();
        Vector3* thisAcc = getAcceleration();
        Epoch* thisEp = getEpoch();
        Covariance* thisCov = getCovariance();
        char* thisBytes = getBytes();

        if (getByteArraySize() != source.getByteArraySize())
        {
            std::cout << "Warning: Cannot insert byte array into population!" << std::endl;
        }

        if (list.getSize() >= source.getSize())
        {
            for(int i = 0; i < list.getSize(); ++i)
            {
                int l = listdata[i];
                if (l < getSize())
                {
                    thisOrbit[l] = orbits[i];
                    thisProps[l] = props[i];
                    thisPos[l] = pos[i];
                    thisVel[l] = vel[i];
                    thisAcc[l] = acc[i];
                    thisEp[l] = ep[i];
                    thisCov[l] = cov[i];
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
        update(DATA_PROPERTIES);
        update(DATA_POSITION);
        update(DATA_VELOCITY);
        update(DATA_ACCELERATION);
        update(DATA_EPOCH);
        update(DATA_COVARIANCE);
        update(DATA_BYTES);
    }

	void Population::remove(int index)
	{
		data->data_acceleration.remove(index);
		data->data_orbit.remove(index);
		data->data_position.remove(index);
		data->data_properties.remove(index);
        data->data_velocity.remove(index);
        data->data_epoch.remove(index);
        data->data_covariance.remove(index);
        data->data_bytes.remove(index*data->byteArraySize, data->byteArraySize);
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
			case DATA_VELOCITY:
				data->data_velocity.update(device);
				break;
            case DATA_POSITION:
				data->data_position.update(device);
				break;
			case DATA_ACCELERATION:
				data->data_acceleration.update(device);
                break;
            case DATA_EPOCH:
                data->data_epoch.update(device);
                break;
            case DATA_COVARIANCE:
                data->data_covariance.update(device);
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

	int Population::getSize() const
	{
		return data->size;
	}

    int Population::getByteArraySize() const
    {
        return data->byteArraySize;
    }

    Host& Population::getHostPointer() const
    {
        return data->host;
    }

    double Population::getLatestEpoch() const
    {
        const double mjd1950 = 2433282.5;
        double latestEpoch = 0.0;
        for (int i=0; i<getSize(); i++)
        {
            double currentEpoch = getEpoch()[i].current_epoch;
            if (currentEpoch < mjd1950)
            {
                return 0.0;
            }
            else {
                latestEpoch = std::max(latestEpoch, currentEpoch);
            }
        }
        return latestEpoch;
    }

    double Population::getEarliestEpoch() const
    {
        const double mjd1950 = 2433282.5;
        double earliestEpoch = 9999999.0;
        for (int i=0; i<getSize(); i++)
        {
            double currentEpoch = getEpoch()[i].current_epoch;
            if (currentEpoch < mjd1950)
            {
                return 0.0;
            }
            else {
                earliestEpoch = std::min(earliestEpoch, currentEpoch);
            }
        }
        return earliestEpoch;
    }

    std::string Population::validate(IndexList& invalidObjects) const
    {
        //0: No object has a current epoch set
        //1: All objects have a current epoch set
        //-1: Some objects have a current epoch set while others do not (bad)
        int epochCheck = 0;
        std::stringstream report;
        for (int i=0; i<data->size; i++)
        {
            bool valid = true;

            // get ID, or -1 if object has no properties set
            int id = -1;

            // check properties
            ObjectProperties props = getObjectProperties(DEVICE_HOST)[i];
            if (data->data_properties.hasData() && !isZero(props))
            {
                id = props.id;
                if (hasNaN(props))
                {
                    report << i << "/" << id << "/Properties: NaN detected" << std::endl;
                    valid = false;
                }
                if (props.drag_coefficient < 1e-32)
                {
                    report << i << "/" << id << "/Properties: Invalid drag coefficient" << std::endl;
                    valid = false;
                }
                if (props.mass < 1e-12)
                {
                    report << i << "/" << id << "/Properties: Invalid mass" << std::endl;
                    valid = false;
                }
                if (props.diameter < 1e-12)
                {
                    report << i << "/" << id << "/Properties: Invalid diameter" << std::endl;
                    valid = false;
                }
                if (props.area_to_mass <= 0.0)
                {
                    report << i << "/" << id << "/Properties: Invalid area to mass ratio" << std::endl;
                    valid = false;
                }
                if (props.reflectivity < 0.0 || props.reflectivity > 2.0)
                {
                    report << i << "/" << id << "/Properties: Invalid reflectivity coefficient" << std::endl;
                    valid = false;
                }
            }

            // check orbit and epoch
            Orbit orbit = getOrbit(DEVICE_HOST)[i];
            Epoch epoch = getEpoch(DEVICE_HOST)[i];
            const double mjd1950 = 2433282.5;
            if (data->data_orbit.hasData() && !isZero(orbit))
            {
                if (hasNaN(orbit))
                {
                    report << i << "/" << id << "/Orbit: NaN detected" << std::endl;
                    valid = false;
                }
                if (orbit.semi_major_axis < 6378.0 && epoch.end_of_life <= 0.0) {
                    report << i << "/" << id << "/Orbit: SMA too small, and object has not been marked as decayed (EOL = 0)" << std::endl;
                    valid = false;
                }
                if (orbit.eccentricity <= 0.0 || orbit.eccentricity >= 1.0)
                {
                    report << i << "/" << id << "/Orbit: Eccentricity not within valid range" << std::endl;
                    valid = false;
                }
                if (orbit.inclination < -2*M_PI || orbit.inclination > 2*M_PI
                    || orbit.raan < -2*M_PI || orbit.raan > 2*M_PI
                    || orbit.arg_of_perigee < -2*M_PI || orbit.arg_of_perigee > 2*M_PI
                    || orbit.mean_anomaly < -2*M_PI || orbit.mean_anomaly > 2*M_PI)
                {
                    report << i << "/" << id << "/Orbit: One or more angles outside radian range" << std::endl;
                    valid = false;
                }
                if (epoch.end_of_life > 0.0 && epoch.beginning_of_life > 0.0 && epoch.end_of_life < epoch.beginning_of_life)
                {
                    report << i << "/" << id << "/Orbit: EOL date precedes BOL date" << std::endl;
                    valid = false;
                }
                if (epoch.current_epoch < mjd1950)
                {
                    if (i == 0) epochCheck = 0;
                    else if (epochCheck == 1) epochCheck = -1;
                }
                else {
                    if (i == 0) epochCheck = 1;
                    else if (epochCheck == 0) epochCheck = -1;
                }
            }

            // check state vector
            Vector3 pos = getPosition(DEVICE_HOST)[i];
            Vector3 vel = getVelocity(DEVICE_HOST)[i];
            if (data->data_position.hasData() && !isZero(pos))
            {
                if (data->data_velocity.hasData() && !isZero(vel))
                {
                    if (length(pos) <= 6378.0 && epoch.end_of_life <= 0.0)
                    {
                        report << i << "/" << id << "/StateVector: Object is inside Earth and has not been marked as decayed (EOL = 0)" << std::endl;
                        valid = false;
                    }
                    if (length(pos) <= 0.0)
                    {
                        report << i << "/" << id << "/StateVector: Invalid velocity" << std::endl;
                        valid = false;
                    }
                }
                else {
                    report << i << "/" << id << "/StateVector: Object has position but no velocity set" << std::endl;
                    valid = false;
                }
            }

            if (isZero(orbit) && isZero(pos) && isZero(vel))
            {
                report << i << "/" << id << "/StateVector: Object has neither state vector nor orbit set" << std::endl;
                valid = false;
            }

            if (!valid) invalidObjects.add(i);
        }

        if (epochCheck == -1) report << "Population: Not all objects have a current epoch set." << std::endl;

        return report.str();
    }

    // adapted from SGP4 reference implementation by D. Vallado e.a.
    // https://celestrak.com/publications/AIAA/2006-6753/
    ErrorCode Population::convertOrbitsToStateVectors()
    {
        if (data->data_orbit.hasData())
        {
            for (int i=0; i<getSize(); i++)
            {
                Orbit o = getOrbit()[i];

                double ta = trueAnomaly(o);
                double argp = o.arg_of_perigee;
                const double arglat = argp + ta;
                double raan = o.raan;
                const double small = 1e-15;

                if (o.eccentricity < small)
                {
                    // circular equatorial
                    if ((o.inclination < small) || (fabs(o.inclination-M_PI) < small))
                    {
                        argp = 0.0;
                        raan = 0.0;
                        ta = o.raan + arglat;
                    }
                    else {
                        // circular inclined
                        argp = 0.0;
                        ta = arglat;
                    }
                }
                else {
                    // elliptical equatorial
                    if ((o.inclination < small) || (fabs(o.inclination-M_PI) < small))
                    {
                        argp = o.raan + o.arg_of_perigee;
                        raan = 0.0;
                    }
                }

                // ----------  form pqw position and velocity vectors ----------
                double p = o.semi_major_axis * (1.0 - pow(o.eccentricity, 2.0));
                const double mu = 398600.4418;
                const double temp = p / (1.0 + o.eccentricity * cos(ta));
                Vector3 r;
				r.x = temp*cos(ta);
				r.y = temp*sin(ta);
				r.z = 0.0;
                if (fabs(p) < 1e-8 ) p = 1e-8;

                Vector3 v;
				v.x = -sin(ta) * sqrt(mu/p);
                v.y = (o.eccentricity + cos(ta)) * sqrt(mu/p);
                v.z = 0.0;

                r = rotateZ(r,-argp);
                r = rotateX(r,-o.inclination);
                r = rotateZ(r,-raan);
                getPosition()[i] = r;

                v = rotateZ(v,-argp);
                v = rotateX(v,-o.inclination);
                v = rotateZ(v,-raan);
                getVelocity()[i] = v;
            }

            update(DATA_POSITION);
            update(DATA_VELOCITY);
            return SUCCESS;
        }
        else return INVALID_DATA;
    }

    // adapted from SGP4 reference implementation by D. Vallado e.a.
    // https://celestrak.com/publications/AIAA/2006-6753/
    ErrorCode Population::convertStateVectorsToOrbits()
    {
        ErrorCode e = SUCCESS;
        if (data->data_position.hasData() && data->data_velocity.hasData())
        {
            enum typeorbit_t {
                CIRCULAR_EQUATORIAL,
                CIRCULAR_INCLINED,
                ELLIPTICAL_EQUATORIAL,
                ELLIPTICAL_INCLINED
            } typeorbit;

            const double twopi = 2.0 * M_PI;
            const double halfpi = 0.5 * M_PI;
            const double small = 1e-15;
            const double mu = 398600.4418;
            const double nan = std::numeric_limits<double>::quiet_NaN();

            for (int i=0; i<getSize(); i++)
            {
                Vector3 r = getPosition()[i];
                Vector3 v = getVelocity()[i];
                Orbit o;

                const double magr = length(r);
                const double magv = length(v);

                Vector3 hbar = cross(r,v);
                const double magh = length(hbar);

                if (magh > small)
                {
                    // ------------------  find h n and e vectors   ----------------
                    Vector3 nbar;
					nbar.x = -hbar.y;
					nbar.y = hbar.x;
					nbar.z = 0.0;
                    const double magn = length(nbar);
                    const double c1 = magv*magv - mu/magr;
                    const double rdotv = r * v;
                    Vector3 ebar = (r*c1 - v*rdotv) / mu;
                    o.eccentricity = length(ebar);

                    // ------------  find a e and semi-latus rectum   ----------
                    const double sme = (magv*magv*0.5) - (mu / magr);
                    if (fabs(sme) > small)
                    {
                        o.semi_major_axis = -mu / (2.0 *sme);
                    }
                    else {
                        o.semi_major_axis = std::numeric_limits<double>::infinity();
                        e = INVALID_DATA;
                    }
                    //const double p = magh*magh / mu;

                    // -----------------  find inclination   -------------------
                    const double hk = hbar.z / magh;
                    o.inclination = acos(hk);

                    // --------  determine type of orbit for later use  --------
                    // ------ elliptical, parabolic, hyperbolic inclined -------
                    typeorbit = ELLIPTICAL_INCLINED;
                    if (o.eccentricity < small)
                    {
                        // ----------------  circular equatorial ---------------
                        if ((o.inclination < small) || (fabs(o.inclination - M_PI) < small))
                            typeorbit = CIRCULAR_EQUATORIAL;
                        else
                            // --------------  circular inclined ---------------
                            typeorbit = CIRCULAR_INCLINED;
                    }
                    else {
                        // - elliptical, parabolic, hyperbolic equatorial --
                        if ((o.inclination < small) || (fabs(o.inclination - M_PI) < small))
                            typeorbit = ELLIPTICAL_EQUATORIAL;
                    }

                    // ----------  find longitude of ascending node ------------
                    if (magn > small)
                    {
                        double temp = nbar.x / magn;
                        if (temp > 1.0) temp = 1.0;
                        else if (temp < -1.0) temp = -1.0;
                        o.raan = acos(temp);
                        if (nbar.y < 0.0) o.raan = twopi - o.raan;
                    }
                    else {
                        o.raan = nan;
                        e = INVALID_DATA;
                    }

                    // ---------------- find argument of perigee ---------------
                    if (typeorbit == ELLIPTICAL_INCLINED)
                    {
                        o.arg_of_perigee = angle(nbar,ebar);
                        if (ebar.z < 0.0) o.arg_of_perigee = twopi - o.arg_of_perigee;
                    }
                    else {
                        o.arg_of_perigee = nan;
                        e = INVALID_DATA;
                    }

                    // ------------  find true anomaly at epoch    -------------
                    double nu = nan;
                    if (typeorbit == ELLIPTICAL_EQUATORIAL || typeorbit == ELLIPTICAL_INCLINED)
                    {
                        nu = angle(ebar,r);
                        if (rdotv < 0.0) nu = twopi - nu;
                    }

                    // ----  find argument of latitude - circular inclined -----
                    double arglat = nan;
                    if (typeorbit == CIRCULAR_INCLINED)
                    {
                        arglat = angle(nbar,r);
                        if (r.z < 0.0) arglat = twopi - arglat;
                        o.mean_anomaly = arglat;
                    }

                    // -- find longitude of perigee - elliptical equatorial ----
                    double lonper = nan;
                    if ((o.eccentricity > small) && (typeorbit == ELLIPTICAL_EQUATORIAL))
                    {
                        double temp = ebar.x / o.eccentricity;
                        if (temp > 1.0) temp = 1.0;
                        else if (temp < -1.0) temp = -1.0;
                        lonper = acos(temp);
                        if (ebar.y < 0.0) lonper = twopi - lonper;
                        if (o.inclination > halfpi) lonper = twopi - lonper;
                    }

                    // -------- find true longitude - circular equatorial ------
                    double truelon = nan;
                    if ((magr>small) && (typeorbit == CIRCULAR_EQUATORIAL))
                    {
                        double temp = r.x / magr;
                        if (temp > 1.0) temp = 1.0;
                        else if (temp < -1.0) temp = -1.0;
                        truelon = acos(temp);
                        if (r.y < 0.0) truelon = twopi - truelon;
                        if (o.inclination > halfpi) truelon = twopi - truelon;
                        o.mean_anomaly = truelon;
                    }

                    // ------------ find mean anomaly for all orbits -----------
                    if (typeorbit == ELLIPTICAL_EQUATORIAL || typeorbit == ELLIPTICAL_INCLINED)
                    {
                        // get mean anomaly from true anomaly
                        const double ecc = o.eccentricity;
                        double ea = std::numeric_limits<double>::infinity();
                        double ma = std::numeric_limits<double>::infinity();

                        if (fabs(ecc) < small)
                        {
                            // circular
                            ea = nu;
                            ma = nu;
                        }
                        else if (ecc < 1.0 - small)
                        {
                            // elliptical
                            const double sine = (sqrt(1.0 - pow(ecc,2.0)) * sin(nu)) / (1.0 + ecc*cos(nu));
                            const double cose = (ecc + cos(nu)) / (1.0 + ecc*cos(nu));
                            ea = atan2(sine, cose);
                            ma = ea - ecc*sin(ea);
                        }
                        else if (ecc > 1.0 + small)
                        {
                            // hyperbolic
                            if ((ecc > 1.0) && (fabs(nu) + 0.00001 < M_PI - acos(1.0 / ecc)))
                            {
                                const double sine = (sqrt(pow(ecc, 2.0) - 1.0) * sin(nu)) / (1.0 + ecc*cos(nu));
								//ea = asinh(sine);
								//Unfortunately, some compilers don't implement all standard functions.
								//Yes, I'm looking at you, Visual Studio!
								ea = log(sine + sqrt(1+pow(sine,2.0)));
                                ma = ecc*sinh(ea) - ea;
                            }
                        }
                        else if (fabs(nu) < 168.0*M_PI / 180.0)
                        {
                            // parabolic
                            ea = tan(nu*0.5);
                            ma = ea + pow(ea,3.0)/3.0;
                        }

                        if (ecc < 1.0)
                        {
                            ma = fmod(ma, 2.0 * M_PI);
                            if (ma < 0.0) ma += 2.0*M_PI;
                        }

                        o.mean_anomaly = ma;
                    }
                }
                else {
                    o.semi_major_axis = nan;
                    o.eccentricity = nan;
                    o.inclination = nan;
                    o.raan = nan;
                    o.arg_of_perigee = nan;
                    o.mean_anomaly = nan;
                    e = INVALID_DATA;
                }
                getOrbit()[i] = o;
            }
            update(DATA_ORBIT);
            return e;
        }
        else return INVALID_DATA;
    }

}
