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
#ifndef OPI_DATA_H
#define OPI_DATA_H
#include "opi_common.h"
#include "opi_error.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
#include <string>
namespace OPI
{
	class ObjectRawData;
	class Host;
	class ObjectProperties;
	class Vector3;
	class IndexPair;
	class IndexList;

	/*! \brief This class contains all parameters required for processing orbital objects.
	 * \ingroup CPP_API_GROUP
	 *
	 * When a Population object is created with a specific size, arrays of the types Orbit,
	 * ObjectProperties and ObjectStatus are initialized with that size. These arrays are empty
     * and must be filled with actual data by the host or the plugin. The "bytes" array can be
     * used to store arbitrary, per-object information. Its per-object size (default: 1 byte)
     * can be adjusted using the resizeByteArray function.
	 */
	class OPI_API_EXPORT Population
	{
		public:
            /**
             * @brief Population Creates a new empty Population, optionally with a given size.
             * @param host A pointer to the OPI Host that this Population is intended for.
             * @param size The number of elements of the Population. Defaults to zero if unset.
             */
			Population(Host& host, int size = 0);

            /**
             * @brief Population Copy constructor
             *
             * This function creates a deep copy of a Population on the host. Device data
             * will be downloaded as part of this operation. The copy does not contain any
             * device data so synchronization will happen again in the other direction as soon
             * as the copy is requested on the device. This function may therefore impact
             * the performance of CUDA/OpenCL propagators when used in every propagation step.
             *
             * @param source The Population to be copied from.
             */
            Population(const Population& source);

            /**
             * @brief Population Copy constructor (indexed copy)
             *
             * Creates a selective deep copy of a Population on the host. Only
             * elements that appear in the given index list are copied to the new Population.
             * Like the full copy, the operation happens entirely on the host so device
             * data is synchronized. This may be even slower than the full copy in some cases.
             *
             * @param source The Population to be copied from.
             * @param list An IndexList containing the indices of the elements of the source
             * Population that should be copied.
             */
            Population(const Population& source, IndexList &list);

            /**
             * @brief Destructor. Cleans up host and device memory.
             */
			~Population();

            /**
             * @brief resize Sets the number of elements of the Population.
             *
             * @param size The new number of elements the Population should contain.
             * @param byteArraySize The per-object size of the byte array that can be queried
             * with the getBytes() function. Defaults to 1 if unset.
             */
            void resize(int size, int byteArraySize = 1);

            /**
             * @brief resizeByteArray Set the per-object size of the byte array.
             *
             * Every object has a byte buffer that can be used to store arbitrary per-object
             * information. This function sets the number of bytes that are available to each
             * object in the Population.
             * @param size The new byte array size.
             */
            void resizeByteArray(int size);

            /**
             * @brief getSize Returns the number of elements in the Population.
             * @return Number of elements.
             */
            int getSize() const;

            //! Returns the per-object size of the byte buffer

            /**
             * @brief getByteArraySize Returns the per-object size of the byte array.
             * @return Number of bytes every object can store in the byte array.
             */
            int getByteArraySize() const;

            /**
             * @brief getLastPropagatorName Returns the name of the last plugin the Population
             * was propagated with.
             * @return The Propagator name as defined by the plugin last used on this Population.
             */
            std::string getLastPropagatorName() const;

            /**
             * @brief getObjectName Returns the name of the given object.
             * Names are host-only attributes and do not get synchronized to the GPU.
             * @return The object name as a string. If no name has been set or the index is out of
             * range, an empty string is returned.
             */
            std::string getObjectName(int index);

            /**
             * @brief setObjectName Set the name of the given object.
             * Names are host-only attributes and do not get synchronized to the GPU.
             * @param index The index of the object.
             * @param name The new name for the object.
             */
            void setObjectName(int index, std::string name);

            /**
             * @brief setLastPropagatorName Set the name of the last Propagator the population was
             * propagated with.
             *
             * This is done automatically by OPI on successful propagation
             * and should not require any extra effort from the plugin author.
             * @param propagatorName The name of the Propagator as returned by its getName() function.
             */
            void setLastPropagatorName(std::string propagatorName);

            /**
             * @brief convertOrbitsToStateVectors convert the population's orbit information to state vectors.
             *
             * This function can be called after setting orbit data to fill the population's position
             * and velocity vectors by converting the orbits.
             * @return INVALID_DATA if orbit data has not been set, SUCCESS otherwise.
             */
            ErrorCode convertOrbitsToStateVectors();

            /**
             * @brief convertStateVectorsToOrbits convert the population's state vectors to orbits.
             *
             * This function can be called after setting position and velocity vectors to fill the population's
             * orbit data by converting them into orbits.
             * @return INVALID_DATA if position/velocity data has not been set or any of the oprations results
             * in NaN, SUCCESS otherwise.
             */
            ErrorCode convertStateVectorsToOrbits();

            /**
             * @brief insert Insert all elements from another population into this one.
             *
             * The given index list states at which positions the elements are inserted,
             * e.g. if list[0] == 2 then the first element from the source population will
             * be copied into the third element from this population (overwriting the data
             * that may have been stored there). Therefore, the IndexList must have at least
             * as many elements as the source population, and values stored within may not
             * exceed the size of this population.
             * @param source The Population from which the elements are copied.
             * @param list A list of indices into the destination Population.
             */
            void insert(Population& source, IndexList& list);

			//! Removes an object
			void remove(int index);
			//! Removes a number of objects
			void remove(IndexList& list);

			//! Stores the Object Data to disk
			void write(const std::string& filename);
			//! Loads the Object Data from disk
			ErrorCode read(const std::string& filename);

			//! Notify about updates on the specified device
			ErrorCode update(int type, Device device = DEVICE_HOST);

			//! Retrieve the orbital parameters on the specified device
			Orbit* getOrbit(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the object properties on the specified device
			ObjectProperties* getObjectProperties(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the position in cartesian coordinates on the specified device
            Vector3* getPosition(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the velocity in cartesian coordinates on the specified device
			Vector3* getVelocity(Device device = DEVICE_HOST, bool no_sync = false) const;
			//! Retrieve the acceleration in cartesian coordinates on the specified device
            Vector3* getAcceleration(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the covariance information on the specified device
            Covariance* getCovariance(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the arbitrary binary information on the specified device
            char* getBytes(Device device = DEVICE_HOST, bool no_sync = false) const;

            /**
             * @brief sanityCheck Performs various checks on the Population data and generate a debug string.
             *
             * Call on this Population to perform some validity checks of all orbits and properties. Checks
             * include orbit height (must be larger than Earth radius or end-of-life date must be set),
             * eccentricity between 0 and 1, range of angles (within -2PI and +2PI) and sensible values for drag
             * and reflectivity coefficients.
             * This is a host function so the data will be synched to the host when calling this function. It is
             * comparatively slow and should be used for debugging or once after Population data is read from
             * input files.
             * @return Human-readable string that can be printed to the screen or a log file. If no problems are
             * found, an empty string is returned.
             */
			std::string sanityCheck();

        //protected:
            Host& getHostPointer() const;

		private:
			//! Private implementation data
            Pimpl<ObjectRawData> data;
    };
}

#endif
