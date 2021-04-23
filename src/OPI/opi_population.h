#ifndef OPI_DATA_H
#define OPI_DATA_H
#include "opi_common.h"
#include "opi_error.h"
#include "opi_datatypes.h"
#include "opi_pimpl_helper.h"
#include "opi_perturbations.h"
#include <string>

/* Revision number of the OPI data file format, stored for backwards compatibility
 * 001 - Initial value for OPI-2019
 * 002 - Added description field
 * 003 - Added reference frame
 * 004 - Added original_epoch field
 * 005 - Added initial_epoch field
 * 006 - Convert epochs to new JulianDay type
 */
#define OPI_DATA_REVISION_NUMBER 6

namespace OPI
{
    struct ObjectRawData;
    class Host;
    struct ObjectProperties;
    struct Vector3;
    struct IndexPair;
    class IndexList;
    class Perturbations;

    /*! \brief This class contains all parameters required for processing orbital objects.
     * \ingroup CPP_API_GROUP
     *
     * When a Population object is created with a specific size, arrays of the types Orbit,
     * ObjectProperties and ObjectStatus are initialized with that size. These arrays are empty
     * and must be filled with actual data by the host or the plugin. The "bytes" array can be
     * used to store arbitrary, per-object information. Its per-object size (default: 1 byte)
     * can be adjusted using the resizeByteArray function.
     */
    class Population
    {
        public:
            /**
             * @brief Population Creates a new empty Population, optionally with a given size.
             * @param host A pointer to the OPI Host that this Population is intended for.
             * @param size The number of elements of the Population. Defaults to zero if unset.
             */
            OPI_API_EXPORT Population(Host& host, int size = 0);

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
            OPI_API_EXPORT Population(const Population& source);

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
            OPI_API_EXPORT Population(Population& source, IndexList &list);

            /**
             * @brief Destructor. Cleans up host and device memory.
             */
            OPI_API_EXPORT ~Population();

            /**
             * @brief operator = Copy assignment operator for the Population.
             * @param other The population to assign data from.
             * @return A copy of the other population.
             */
            OPI_API_EXPORT Population& operator=(const Population& other);

            OPI_API_EXPORT Population& operator+=(const Population& other);
            OPI_API_EXPORT Population& operator+=(const Perturbations& delta);
            OPI_API_EXPORT Population operator+(const Population& other);
            OPI_API_EXPORT Population operator+(const Perturbations& delta);

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
            OPI_API_EXPORT const char* getLastPropagatorName() const;

            /**
             * @brief getDescription Get the population's description string.
             * @return The string containing the description.
             */
            OPI_API_EXPORT const char* getDescription() const;

            /**
             * @brief getReferenceFrame Get the population's reference frame.
             * @return The reference frame information.
             */
            OPI_API_EXPORT ReferenceFrame getReferenceFrame() const;

            /**
             * @brief getObjectName Returns the name of the given object.
             * Names are host-only attributes and do not get synchronized to the GPU.
             * @return The object name as a string. If no name has been set or the index is out of
             * range, an empty string is returned.
             */
            OPI_API_EXPORT const char* getObjectName(int index) const;

            /**
             * @brief getEarliestEpoch Find the earliest epoch from the object's current_epoch fields.
             * @return 0.0 if any object has an invalid (i.e. lower than Jan 1st, 1950) current epoch set,
             * otherwise the earliest Julian date found.
             */
            OPI_API_EXPORT JulianDay getEarliestEpoch() const;

            /**
             * @brief getLatestEpoch Find the latest epoch from the object's current_epoch fields.
             * @return 0.0 if any object has an invalid (i.e. lower than Jan 1st, 1950) current epoch set,
             * otherwise the latest Julian date found.
             */
            OPI_API_EXPORT JulianDay getLatestEpoch() const;

            //! Retrieve an object's index based on the ID given in its ObjectProperties field.
            OPI_API_EXPORT int findByID(int id) const;

            /**
             * @brief markedAsDeorbited Checks whether an object has been marked as deorbited (EOL >= current epoch)
             * @param index Index of object to check for
             * @return true if object has been marked as deorbited
             */
            OPI_API_EXPORT bool markedAsDeorbited(int index) const;

            /**
             * @brief setObjectName Set the name of the given object.
             * Names are host-only attributes and do not get synchronized to the GPU.
             * @param index The index of the object.
             * @param name The new name for the object.
             */
            OPI_API_EXPORT void setObjectName(int index, const char* name);

            /**
             * @brief setLastPropagatorName Set the name of the last Propagator the population was
             * propagated with.
             *
             * This is done automatically by OPI on successful propagation
             * and should not require any extra effort from the plugin author.
             * @param propagatorName The name of the Propagator as returned by its getName() function.
             */
            OPI_API_EXPORT void setLastPropagatorName(const char* propagatorName);

            /**
             * @brief setDescription Set a description for the population.
             *
             * This can be displayed to the user and contain information about how and from what
             * sources it was generated, coordinate system the objects are in, valid date range, etc.
             * @param propagatorName The description string.
             */
            OPI_API_EXPORT void setDescription(const char* description);

            /**
             * @brief setReferenceFrame Set the reference frame this population's coordinates are in.
             *
             * May be used by the propagator to determine whether this population can be used or
             * needs to be converted.
             * @param referenceFrame The reference frame this population is in.
             */
            OPI_API_EXPORT void setReferenceFrame(const ReferenceFrame referenceFrame);

            /**
             * @brief convertOrbitsToStateVectors convert the population's orbit information to state vectors.
             *
             * This function can be called after setting orbit data to fill the population's position
             * and velocity vectors by converting the orbits.
             * @return INVALID_DATA if orbit data has not been set, SUCCESS otherwise.
             */
            OPI_API_EXPORT ErrorCode convertOrbitsToStateVectors();

            /**
             * @brief convertStateVectorsToOrbits convert the population's state vectors to orbits.
             *
             * This function can be called after setting position and velocity vectors to fill the population's
             * orbit data by converting them into orbits.
             * @return INVALID_DATA if position/velocity data has not been set or any of the oprations results
             * in NaN, SUCCESS otherwise.
             */
            OPI_API_EXPORT ErrorCode convertStateVectorsToOrbits();

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
            OPI_API_EXPORT void insert(Population& source, IndexList& list);

            //! Removes an object
            OPI_API_EXPORT void remove(int index);
            //! Removes a number of objects
            OPI_API_EXPORT void remove(IndexList& list);

            //! Stores the Object Data to disk
            OPI_API_EXPORT void write(const char* filename) const;
            //! Loads the Object Data from disk
            OPI_API_EXPORT ErrorCode read(const char* filename);

            //! Stores the Object Data as a JSON file. Does not include the byte array.
            OPI_API_EXPORT void writeJSON(const char* filename) const;

            //! Notify about updates on the specified device
            OPI_API_EXPORT ErrorCode update(int type, Device device = DEVICE_HOST);

            //! Retrieve the orbital parameters on the specified device
            OPI_API_EXPORT Orbit* getOrbit(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the object properties on the specified device
            OPI_API_EXPORT ObjectProperties* getObjectProperties(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the position in cartesian coordinates on the specified device
            OPI_API_EXPORT Vector3* getPosition(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the velocity in cartesian coordinates on the specified device
            OPI_API_EXPORT Vector3* getVelocity(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the acceleration in cartesian coordinates on the specified device
            OPI_API_EXPORT Vector3* getAcceleration(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve epoch information on the specified device
            OPI_API_EXPORT Epoch* getEpoch(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the covariance information on the specified device
            OPI_API_EXPORT Covariance* getCovariance(Device device = DEVICE_HOST, bool no_sync = false) const;
            //! Retrieve the arbitrary binary information on the specified device
            OPI_API_EXPORT char* getBytes(Device device = DEVICE_HOST, bool no_sync = false) const;

            /**
             * @brief validate Performs various checks on the Population data and generate a debug string.
             *
             * Call on this Population to perform some validity checks of all orbits and properties. Checks
             * include orbit height (must be larger than Earth radius or end-of-life date must be set),
             * eccentricity between 0 and 1, range of angles (within -2PI and +2PI) and sensible values for drag
             * and reflectivity coefficients.
             * This is a host function so the data will be synched to the host when calling this function. It is
             * comparatively slow and should be used for debugging or once after Population data is read from
             * input files.
             * @param invalidObjects an IndexList to which indices of invalid objects will be added.
             * @return Human-readable string that can be printed to the screen or a log file. If no problems are
             * found, an empty string is returned.
             */
            OPI_API_EXPORT std::string validate(IndexList& invalidObjects) const;

        //protected:
            OPI_API_EXPORT Host& getHostPointer() const;

        private:
            //! Private implementation data
            Pimpl<ObjectRawData> data;

            void rebuildNoradIndex();
    };
}

#endif
