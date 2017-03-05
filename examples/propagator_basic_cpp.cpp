#include <cuda.h>
#include "OPI/opi_cpp.h"

// some plugin information
#define OPI_PLUGIN_NAME "BasicCPP"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "Basic Mean Motion Converter - C++ version"

// the plugin version
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0

// Basic propagator that calculates cartesian position and unperturbed mean motion.
// This is the C++ version. There are equivalent CUDA and OpenCL plugins in the examples folder.
class BasicCPP: public OPI::Propagator
{
    public:
        BasicCPP(OPI::Host& host)
        {
            baseDay = 0;
        }

        virtual ~BasicCPP()
        {

        }

        virtual OPI::ErrorCode runPropagation(OPI::Population& data, double julian_day, float dt )
        {
            // In this simple example, we don't have to fiddle with Julian dates. Instead, we'll just
            // look at the seconds that have elapsed since the first call of the propagator. The first
            // time runPropagation() is called, the given day is saved and then subtracted from the
            // following days. The remainder is converted to seconds and passed to the CUDA kernel.
            if (baseDay == 0) baseDay = julian_day;
            float seconds = (julian_day-baseDay)*86400.0 + dt;

            OPI::Orbit* orbit = data.getOrbit(OPI::DEVICE_HOST);
            OPI::Vector3* position = data.getCartesianPosition(OPI::DEVICE_HOST);

            // Call the propagation function.
            cpp_propagate(orbit, position, seconds, data.getSize());

            // The kernel writes to the Population's position and orbit vectors, so
            // these two have to be marked for updated values on the host device.
            data.update(OPI::DATA_CARTESIAN, OPI::DEVICE_HOST);
            data.update(OPI::DATA_ORBIT, OPI::DEVICE_HOST);

            return OPI::SUCCESS;
        }

        // Saving a member variable like baseDay in the propagator can lead to problems because
        // the host might change the propagation times or even the entire population without
        // notice. Therefore, plugin authors need to make sure that at least when disabling
        // and subsquently enabling the propagator, hosts can expect the propagator to
        // reset to its initial state.
        virtual OPI::ErrorCode runEnable()
        {
            baseDay = 0;
            return OPI::SUCCESS;
        }

        virtual OPI::ErrorCode runDisable()
        {
            baseDay = 0;
            return OPI::SUCCESS;
        }

        // Theoretically, the algorithm can handle backward propagation,
        // but the simplified handling of the input time cannot. Therefore, we'll return false
        // in this function.
        bool backwardPropagation()
        {
            return false;
        }

        // This propagator returns a position vector.
        bool cartesianCoordinates()
        {
            return true;
        }

        // This plugin does not require CUDA.
        int requiresCUDA()
        {
            return 0;
        }

        // This plugin does not require OpenCL.
        int requiresOpenCL()
        {
            return 0;
        }

        // This plugin is written for OPI version 1.0.
        int minimumOPIVersionRequired()
        {
            return 1;
        }

    private:
        double baseDay;

        // Auxiliary function that iteratively converts mean anomaly to eccentric anomaly.
        float mean2eccentric(float meanAnomaly, float eccentricity)
        {
            float eccentricAnomaly = meanAnomaly;
            int maxloop = 5;
            float fcte, fctes;

            for (int i=0; i<maxloop; i++) {
                fcte  = eccentricAnomaly - eccentricity * sin(eccentricAnomaly) - meanAnomaly;
                fctes = 1.0 - eccentricity * cos(eccentricAnomaly);
                eccentricAnomaly -= fcte/fctes;
            }
            return eccentricAnomaly;
        }

        // Function that does the actual transformations. Equivalent to the basic CUDA example.
        void cpp_propagate(OPI::Orbit* orbit, OPI::Vector3* position, float seconds, int size)
        {
            // Loop over all objects in the population.
            for (int i=0; i<size; i++)
            {
                // Store orbit data from the object this kernel is responsible for.
                float sma = orbit[i].semi_major_axis;
                float ecc = orbit[i].eccentricity;
                float inc = orbit[i].inclination;
                float raan = orbit[i].raan;
                float aop = orbit[i].arg_of_perigee;
                float phi = orbit[i].mean_anomaly;

                // Define some auxiliary constants.
                float PI = 3.1415926f;
                float RMUE = 398600.5f;
                float EPSILON = 0.00001f;

                // Confine the input time to the object's orbit period.
                float orbit_period = 2.0f * PI * sqrt(powf(sma,3.0f) / RMUE);
                float t = fmod(seconds, orbit_period);

                // Calculate the mean anomaly and eccentric anomaly.
                // Note: This disregards the initial mean anomaly given in the Population -
                // avoid this in production plugins.
                float mean_anomaly = fmodf(sqrtf((RMUE * t * t) / powf(sma,3.0f)), 2.0f*PI);
                float excentric_anomaly = mean2eccentric(mean_anomaly, ecc);

                // Convert eccentric anomaly to true anomaly.
                float sin_ea = sin(excentric_anomaly/2.0f);
                float cos_ea = cos(excentric_anomaly/2.0f);
                float true_anomaly = 2.0f * atan(sqrtf((1.0f + ecc)/(1.0f - ecc)) * sin_ea/cos_ea);

                // Based on the true anomaly, calculate cartesian object coordinates.
                float u = true_anomaly + aop;
                OPI::Vector3 w;
                w.x = cos(u) * cos(raan) - sin(u) * sin(raan) * cos(inc);
                w.y = cos(u) * sin(raan) + sin(u) * cos(raan) * cos(inc);
                w.z = sin(u) * sin(inc);

                float p = sma * (1.0f - powf(ecc,2.0f));
                float arg = 1.0f + (ecc * cos(true_anomaly));
                float r = p / EPSILON;
                if (arg > EPSILON) r = p / arg;

                // Write the position vector into the OPI::Population array.
                position[i].x = w.x*r;
                position[i].y = w.y*r;
                position[i].z = w.z*r;

                // Finally, also write back the new mean anomaly into the orbit.
                orbit[i].mean_anomaly = mean_anomaly;
            }
        }

};

#define OPI_IMPLEMENT_CPP_PROPAGATOR BasicCPP

#include "OPI/opi_implement_plugin.h"
