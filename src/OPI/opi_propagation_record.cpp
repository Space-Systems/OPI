#include "opi_propagation_record.h"
#include "opi_datatypes.h"
#include "opi_logger.h"

namespace OPI
{
    PropagationRecord::PropagationRecord(Population* population)
    {
        initSwitches();
        populationPointer = population;
    }

    PropagationRecord::PropagationRecord(PropagationRecord& source)
    {
        initSwitches();
        populationPointer = source.populationPointer;
        fixed = source.fixed;
        autoRecord = source.autoRecord;
        recordSwitch = source.recordSwitch;
        epochRecord = source.epochRecord;
        dataRecord = source.dataRecord;
    }

    PropagationRecord::~PropagationRecord()
    {
        populationPointer = nullptr;
    }

    void PropagationRecord::initSwitches()
    {
        fixed = false;
        autoRecord = true;
        for (RecordType r : recordTypeIterator())
        {
            recordSwitch[r] = true;
        }
    }

    void PropagationRecord::ensureManualMode()
    {
        if (autoRecord)
        {
            for (RecordType r : recordTypeIterator())
            {
                recordSwitch[r] = false;
            }
            autoRecord = false;
        }
    }

    void PropagationRecord::addRecordType(RecordType type)
    {
        ensureManualMode();
        if (!fixed) recordSwitch[type] = true;
        else Logger::out(0) << "Cannot add more record types after a snapshot has been taken. Please clear the record first." << std::endl;
    }

    void PropagationRecord::takeSnapshot()
    {
        if (populationPointer != nullptr)
        {
            for (int i=0; i<populationPointer->getSize(); i++)
            {
                takeSnapshot(i);
            }
        }
    }

    void PropagationRecord::takeSnapshot(int objectIndex)
    {
        if (populationPointer != nullptr)
        {
            if (populationPointer->getSize() > objectIndex)
            {
                fixed = true;
                JulianDay epoch = populationPointer->getEpoch()[objectIndex].current_epoch;

                // Always record epoch, regardless of switch
                epochRecord[objectIndex].push_back(epoch);
                dataRecord[objectIndex][REC_EPOCH].push_back(toDouble(epoch));

                // Record all the values that are checked
                if (recordSwitch[REC_POSITION] || recordSwitch[REC_POS_X]) dataRecord[objectIndex][REC_POS_X].push_back(populationPointer->getPosition()[objectIndex].x);
                if (recordSwitch[REC_POSITION] || recordSwitch[REC_POS_Y]) dataRecord[objectIndex][REC_POS_Y].push_back(populationPointer->getPosition()[objectIndex].y);
                if (recordSwitch[REC_POSITION] || recordSwitch[REC_POS_Z]) dataRecord[objectIndex][REC_POS_Z].push_back(populationPointer->getPosition()[objectIndex].z);
                if (recordSwitch[REC_VELOCITY] || recordSwitch[REC_VEL_X]) dataRecord[objectIndex][REC_VEL_X].push_back(populationPointer->getVelocity()[objectIndex].x);
                if (recordSwitch[REC_VELOCITY] || recordSwitch[REC_VEL_Y]) dataRecord[objectIndex][REC_VEL_Y].push_back(populationPointer->getVelocity()[objectIndex].y);
                if (recordSwitch[REC_VELOCITY] || recordSwitch[REC_VEL_Z]) dataRecord[objectIndex][REC_VEL_Z].push_back(populationPointer->getVelocity()[objectIndex].z);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_SMA]) dataRecord[objectIndex][REC_ORB_SMA].push_back(populationPointer->getOrbit()[objectIndex].semi_major_axis);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_ECC]) dataRecord[objectIndex][REC_ORB_ECC].push_back(populationPointer->getOrbit()[objectIndex].eccentricity);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_INC]) dataRecord[objectIndex][REC_ORB_INC].push_back(populationPointer->getOrbit()[objectIndex].inclination);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_RAAN]) dataRecord[objectIndex][REC_ORB_RAAN].push_back(populationPointer->getOrbit()[objectIndex].raan);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_AOP]) dataRecord[objectIndex][REC_ORB_AOP].push_back(populationPointer->getOrbit()[objectIndex].arg_of_perigee);
                if (recordSwitch[REC_ORBIT] || recordSwitch[REC_ORB_MA]) dataRecord[objectIndex][REC_ORB_MA].push_back(populationPointer->getOrbit()[objectIndex].mean_anomaly);
                if (recordSwitch[REC_PROPERTIES] || recordSwitch[REC_PROP_MASS]) dataRecord[objectIndex][REC_PROP_MASS].push_back(populationPointer->getObjectProperties()[objectIndex].mass);
                if (recordSwitch[REC_PROPERTIES] || recordSwitch[REC_PROP_DIA]) dataRecord[objectIndex][REC_PROP_DIA].push_back(populationPointer->getObjectProperties()[objectIndex].diameter);
                if (recordSwitch[REC_PROPERTIES] || recordSwitch[REC_PROP_A2M]) dataRecord[objectIndex][REC_PROP_A2M].push_back(populationPointer->getObjectProperties()[objectIndex].area_to_mass);
                if (recordSwitch[REC_PROPERTIES] || recordSwitch[REC_PROP_CD]) dataRecord[objectIndex][REC_PROP_CD].push_back(populationPointer->getObjectProperties()[objectIndex].drag_coefficient);
                if (recordSwitch[REC_PROPERTIES] || recordSwitch[REC_PROP_CR]) dataRecord[objectIndex][REC_PROP_CR].push_back(populationPointer->getObjectProperties()[objectIndex].reflectivity);
                if (recordSwitch[REC_ACCELERATION] || recordSwitch[REC_ACC_X]) dataRecord[objectIndex][REC_ACC_X].push_back(populationPointer->getAcceleration()[objectIndex].x);
                if (recordSwitch[REC_ACCELERATION] || recordSwitch[REC_ACC_Y]) dataRecord[objectIndex][REC_ACC_Y].push_back(populationPointer->getAcceleration()[objectIndex].y);
                if (recordSwitch[REC_ACCELERATION] || recordSwitch[REC_ACC_Z]) dataRecord[objectIndex][REC_ACC_Z].push_back(populationPointer->getAcceleration()[objectIndex].z);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K1_K1]) dataRecord[objectIndex][REC_COV_K1_K1].push_back(populationPointer->getCovariance()[objectIndex].k1_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K2_K1]) dataRecord[objectIndex][REC_COV_K2_K1].push_back(populationPointer->getCovariance()[objectIndex].k2_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K2_K2]) dataRecord[objectIndex][REC_COV_K2_K2].push_back(populationPointer->getCovariance()[objectIndex].k2_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K3_K1]) dataRecord[objectIndex][REC_COV_K3_K1].push_back(populationPointer->getCovariance()[objectIndex].k3_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K3_K2]) dataRecord[objectIndex][REC_COV_K3_K2].push_back(populationPointer->getCovariance()[objectIndex].k3_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K3_K3]) dataRecord[objectIndex][REC_COV_K3_K3].push_back(populationPointer->getCovariance()[objectIndex].k3_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K4_K1]) dataRecord[objectIndex][REC_COV_K4_K1].push_back(populationPointer->getCovariance()[objectIndex].k4_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K4_K2]) dataRecord[objectIndex][REC_COV_K4_K2].push_back(populationPointer->getCovariance()[objectIndex].k4_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K4_K3]) dataRecord[objectIndex][REC_COV_K4_K3].push_back(populationPointer->getCovariance()[objectIndex].k4_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K4_K4]) dataRecord[objectIndex][REC_COV_K4_K4].push_back(populationPointer->getCovariance()[objectIndex].k4_k4);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K5_K1]) dataRecord[objectIndex][REC_COV_K5_K1].push_back(populationPointer->getCovariance()[objectIndex].k5_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K5_K2]) dataRecord[objectIndex][REC_COV_K5_K2].push_back(populationPointer->getCovariance()[objectIndex].k5_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K5_K3]) dataRecord[objectIndex][REC_COV_K5_K3].push_back(populationPointer->getCovariance()[objectIndex].k5_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K5_K4]) dataRecord[objectIndex][REC_COV_K5_K4].push_back(populationPointer->getCovariance()[objectIndex].k5_k4);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K5_K5]) dataRecord[objectIndex][REC_COV_K5_K5].push_back(populationPointer->getCovariance()[objectIndex].k5_k5);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K1]) dataRecord[objectIndex][REC_COV_K6_K1].push_back(populationPointer->getCovariance()[objectIndex].k6_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K2]) dataRecord[objectIndex][REC_COV_K6_K2].push_back(populationPointer->getCovariance()[objectIndex].k6_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K3]) dataRecord[objectIndex][REC_COV_K6_K3].push_back(populationPointer->getCovariance()[objectIndex].k6_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K4]) dataRecord[objectIndex][REC_COV_K6_K4].push_back(populationPointer->getCovariance()[objectIndex].k6_k4);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K5]) dataRecord[objectIndex][REC_COV_K6_K5].push_back(populationPointer->getCovariance()[objectIndex].k6_k5);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_K6_K6]) dataRecord[objectIndex][REC_COV_K6_K6].push_back(populationPointer->getCovariance()[objectIndex].k6_k6);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K1]) dataRecord[objectIndex][REC_COV_D1_K1].push_back(populationPointer->getCovariance()[objectIndex].d1_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K2]) dataRecord[objectIndex][REC_COV_D1_K2].push_back(populationPointer->getCovariance()[objectIndex].d1_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K3]) dataRecord[objectIndex][REC_COV_D1_K3].push_back(populationPointer->getCovariance()[objectIndex].d1_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K4]) dataRecord[objectIndex][REC_COV_D1_K4].push_back(populationPointer->getCovariance()[objectIndex].d1_k4);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K5]) dataRecord[objectIndex][REC_COV_D1_K5].push_back(populationPointer->getCovariance()[objectIndex].d1_k5);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_K6]) dataRecord[objectIndex][REC_COV_D1_K6].push_back(populationPointer->getCovariance()[objectIndex].d1_k6);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D1_D1]) dataRecord[objectIndex][REC_COV_D1_D1].push_back(populationPointer->getCovariance()[objectIndex].d1_d1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K1]) dataRecord[objectIndex][REC_COV_D2_K1].push_back(populationPointer->getCovariance()[objectIndex].d2_k1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K2]) dataRecord[objectIndex][REC_COV_D2_K2].push_back(populationPointer->getCovariance()[objectIndex].d2_k2);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K3]) dataRecord[objectIndex][REC_COV_D2_K3].push_back(populationPointer->getCovariance()[objectIndex].d2_k3);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K4]) dataRecord[objectIndex][REC_COV_D2_K4].push_back(populationPointer->getCovariance()[objectIndex].d2_k4);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K5]) dataRecord[objectIndex][REC_COV_D2_K5].push_back(populationPointer->getCovariance()[objectIndex].d2_k5);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_K6]) dataRecord[objectIndex][REC_COV_D2_K6].push_back(populationPointer->getCovariance()[objectIndex].d2_k6);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_D1]) dataRecord[objectIndex][REC_COV_D2_D1].push_back(populationPointer->getCovariance()[objectIndex].d2_d1);
                if (recordSwitch[REC_COVARIANCE] || recordSwitch[REC_COV_D2_D2]) dataRecord[objectIndex][REC_COV_D2_D2].push_back(populationPointer->getCovariance()[objectIndex].d2_d2);
            }
            else Logger::out(0) << "Cannot take snapshot - object index " << objectIndex << " does not exist in population!" << std::endl;
        }
        else Logger::out(0) << "Cannot take snapshot - population object not set!" << std::endl;
    }

    unsigned int PropagationRecord::getSampleSize(int objectIndex)
    {
        return (epochRecord.find(objectIndex) == epochRecord.end()) ? 0 : epochRecord[objectIndex].size();
    }

    const JulianDay* PropagationRecord::getEpoch(int objectIndex)
    {
        return (epochRecord.find(objectIndex) == epochRecord.end()) ? nullptr : &(epochRecord[objectIndex])[0];
    }

    const double* PropagationRecord::getSample(int objectIndex, RecordType type)
    {
        if (dataRecord.find(objectIndex) == dataRecord.end())
            return nullptr;
        else if (dataRecord[objectIndex].find(type) == dataRecord[objectIndex].end())
            return nullptr;
        else if (dataRecord[objectIndex][type].size() != getSampleSize(objectIndex))
        {
            Logger::out(0) << "Warning: Inconsistencies detected in population record - returning null pointer." << std::endl;
            return nullptr;
        }
        else
            return &(dataRecord[objectIndex][type])[0];
    }

    void PropagationRecord::clear()
    {
        dataRecord.clear();
        epochRecord.clear();
        fixed = false;
    }

    const std::vector<double> PropagationRecord::getSampleArray(int objectIndex, RecordType type)
    {
        if (dataRecord.find(objectIndex) == dataRecord.end())
            return std::vector<double>(0);
        else if (dataRecord[objectIndex].find(type) == dataRecord[objectIndex].end())
            return std::vector<double>(0);
        else if (dataRecord[objectIndex][type].size() != getSampleSize(objectIndex))
        {
            Logger::out(0) << "Warning: Inconsistencies detected in population record - returning empty sample set." << std::endl;
            return std::vector<double>(getSampleSize(objectIndex), 0.0);
        }
        else
            return dataRecord[objectIndex][type];
    }

    const std::vector<JulianDay> PropagationRecord::getEpochArray(int objectIndex)
    {
        return (epochRecord.find(objectIndex) == epochRecord.end()) ? std::vector<JulianDay>(0) : epochRecord[objectIndex];
    }
}

