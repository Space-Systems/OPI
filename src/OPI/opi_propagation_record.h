#ifndef OPI_PROPAGATION_RECORD_H
#define OPI_PROPAGATION_RECORD_H

#include "opi_common.h"
#include "opi_population.h"
#include "opi_indexlist.h"
#include "opi_types.h"

#include <vector>
#include <map>
#include <type_traits>

namespace OPI
{
    //enum iterator, used for setting the initial switches
    //https://stackoverflow.com/questions/261963/how-can-i-iterate-over-an-enum/26910769
    template < typename C, C beginVal, C endVal>
    class Iterator
    {
        typedef typename std::underlying_type<C>::type val_t;
        int val;
        public:
            Iterator(const C & f) : val(static_cast<val_t>(f)) {}
            Iterator() : val(static_cast<val_t>(beginVal)) {}
            Iterator operator++() {
              ++val;
              return *this;
            }
            C operator*() { return static_cast<C>(val); }
            Iterator begin() { return *this; } //default ctor is good
            Iterator end() {
                static const Iterator endIter=++Iterator(endVal); // cache it
                return endIter;
            }
            bool operator!=(const Iterator& i) { return val != i.val; }
    };

    class PropagationRecord
    {
        public:
            /**
             * @brief Creates a Propagation Record for a given Population.
             *
             * It can be used to keep track of changes in certain values during
             * propagation, e.g. for plotting. To use it, create an instance of
             * this class in your host application after you created the population
             * that should be recorded. After each propagation step, call
             * takeSnapshot() with an optional object index to store the most important
             * values of that object (or all objects) along with their current epoch.
             * By default, the values stored are position, velocity, properties except ID,
             * orbit, acceleration and covariance. To limit the record to specific values,
             * the function addRecordType() can be used.
             *
             * @param population A pointer to the Population that is going to be recorded.
             */
            OPI_API_EXPORT PropagationRecord(Population* population);

            /**
             * @brief Copy constructor.
             * @param source The Propagation Record to be copied from.
             */
            OPI_API_EXPORT PropagationRecord(PropagationRecord& source);

            /**
             * @brief Class destructor.
             */
            OPI_API_EXPORT ~PropagationRecord();

            /**
             * @brief Adds a value to be recorded.
             *
             * This function is used to specify which population values should be recorded.
             * It can be called multiple times to specify multiple types, but calling it at all
             * is entirely optional. If it is not called all values will be recorded by default,
             * otherwise recording will be limited to the types specified.
             * It must be called before the first call to takeSnapshot().
             *
             * @param type Enum value specifying the type of data to be recorded.
             */
            OPI_API_EXPORT void addRecordType(RecordType type);

            /**
             * @brief Records a snapshot for a single object or the entire population.
             *
             */
            OPI_API_EXPORT void takeSnapshot();
            OPI_API_EXPORT void takeSnapshot(int objectIndex);

            OPI_API_EXPORT unsigned int getSampleSize(int objectIndex);
            OPI_API_EXPORT const JulianDay* getEpoch(int objectIndex);
            OPI_API_EXPORT const std::vector<double> getSampleArray(int objectIndex, RecordType type);
            OPI_API_EXPORT const std::vector<JulianDay> getEpochArray(int objectIndex);
            OPI_API_EXPORT const double* getSample(int objectIndex, RecordType type);
            OPI_API_EXPORT void clear();

        private:
            typedef Iterator<RecordType, RecordType::REC_EPOCH, RecordType::REC_COV_D2_D2> recordTypeIterator;
            typedef std::map<RecordType, std::vector<double>> typeMap;
            typedef std::map<int, typeMap> doubleMap;
            typedef std::map<int, std::vector<JulianDay>> epochMap;

            bool autoRecord;
            bool fixed;
            std::map<RecordType, bool> recordSwitch;
            doubleMap dataRecord;
            epochMap epochRecord;

            OPI::Population* populationPointer;

            void initSwitches();
            void ensureManualMode();

    };
}

#endif // PROPAGATIONRECORD_H
