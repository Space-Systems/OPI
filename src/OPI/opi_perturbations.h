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
#ifndef OPI_PERTURBATIONS_H
#define OPI_PERTURBATIONS_H
#include "opi_common.h"
#include "opi_error.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
#include "opi_population.h"
#include <string>
namespace OPI
{
    struct PerturbationRawData;
    class Host;
    struct ObjectProperties;
    struct Vector3;
    struct IndexPair;
    class IndexList;

    /*! \brief This class contains device-synchronizable perturbation information.
     * \ingroup CPP_API_GROUP
     *
     * The purpose of this class is to hold the changes in the object parameters that a
     * Perturbation Module applies to a given population. Like the Population class it can
     * be synchronized to a GPU computing device automatically.
     */
    class Perturbations
    {
        public:
            /**
             * @brief Perturbations Creates a new Perturbations object, optionally with a given size.
             * @param host A pointer to the OPI Host that this instance is intended for.
             * @param size The number of elements. It should match the size of the corresponding Population.
             */
			OPI_API_EXPORT Perturbations(const Population& population);

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
			OPI_API_EXPORT Perturbations(const Perturbations& source);

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
			OPI_API_EXPORT Perturbations(const Perturbations& source, IndexList &list);

            /**
             * @brief Destructor. Cleans up host and device memory.
             */
			OPI_API_EXPORT ~Perturbations();

            /**
             * @brief resize Sets the number of elements of the Population.
             *
             * @param size The new number of elements the Population should contain.
             * @param byteArraySize The per-object size of the byte array that can be queried
             * with the getBytes() function. Defaults to 1 if unset.
             */
			OPI_API_EXPORT void resize(int size, int byteArraySize = 1);

            /**
             * @brief resizeByteArray Set the per-object size of the byte array.
             *
             * Every object has a byte buffer that can be used to store arbitrary per-object
             * information. This function sets the number of bytes that are available to each
             * object in the Population.
             * @param size The new byte array size.
             */
			OPI_API_EXPORT void resizeByteArray(int size);

            /**
             * @brief getSize Returns the number of elements in the Population.
             * @return Number of elements.
             */
			OPI_API_EXPORT int getSize() const;

            //! Returns the per-object size of the byte buffer

            /**
             * @brief getByteArraySize Returns the per-object size of the byte array.
             * @return Number of bytes every object can store in the byte array.
             */
			OPI_API_EXPORT int getByteArraySize() const;

            /**
             * @brief getLastPropagatorName Returns the name of the last plugin the Population
             * was propagated with.
             * @return The Propagator name as defined by the plugin last used on this Population.
             */
			OPI_API_EXPORT std::string getLastPropagatorName() const;

            /**
             * @brief setLastPropagatorName Set the name of the last Propagator the population was
             * propagated with.
             *
             * This is done automatically by OPI on successful propagation
             * and should not require any extra effort from the plugin author.
             * @param propagatorName The name of the Propagator as returned by its getName() function.
             */
			OPI_API_EXPORT void setLastPropagatorName(std::string propagatorName);

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
			OPI_API_EXPORT void insert(Perturbations& source, IndexList& list);

            //! Removes an object
			OPI_API_EXPORT void remove(int index);
            //! Removes a number of objects
			OPI_API_EXPORT void remove(IndexList& list);

            //! Stores the Object Data to disk
			OPI_API_EXPORT void write(const std::string& filename);
            //! Loads the Object Data from disk
			OPI_API_EXPORT ErrorCode read(const std::string& filename);

            //! Notify about updates on the specified device
			OPI_API_EXPORT ErrorCode update(int type, Device device = DEVICE_HOST);

            //! Retrieve the orbital parameters on the specified device
			OPI_API_EXPORT Orbit* getDeltaOrbit(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the position in cartesian coordinates on the specified device
			OPI_API_EXPORT Vector3* getDeltaPosition(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the velocity in cartesian coordinates on the specified device
			OPI_API_EXPORT Vector3* getDeltaVelocity(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the acceleration in cartesian coordinates on the specified device
			OPI_API_EXPORT Vector3* getDeltaAcceleration(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the V matrix on the specified device
			OPI_API_EXPORT VMatrix* getVMatrix(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the arbitrary binary information on the specified device
			OPI_API_EXPORT char* getBytes(Device device = DEVICE_HOST, bool no_sync = false) const;

        protected:
            Host& getHostPointer() const;

        private:
            //! Private implementation data
            Pimpl<PerturbationRawData> data;
    };
}

#endif
