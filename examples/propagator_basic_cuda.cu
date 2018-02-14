#include <cuda.h>
#include "OPI/opi_cpp.h"

// Basic information about the plugin that can be queried by the host.
// OPI_PLUGIN_NAME is the most important as it serves as an identifier
// for the host to request a specific propagator. It should be descriptive
// and closely match the plugin's file name.
#define OPI_PLUGIN_NAME "BasicCUDA"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "Basic Mean Motion Converter - CUDA version"

// Set the version number for the plugin here.
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0


// Auxiliary kernel function that iteratively converts mean anomaly to eccentric anomaly.
__host__ __device__ float mean2eccentric(float meanAnomaly, float eccentricity)
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

// CUDA kernel that does the actual transformations.
// Adapted from https://doi.org/10.5281/zenodo.322256, Listing 6.3
__global__ void kernel_propagate(OPI::Orbit* orbit, OPI::Vector3* position, float seconds, int size)
{
    // Get the kernel ID...
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // ...and make sure it does not exceed the Population size.
    if (idx < size)
    {
        // Store orbit data from the object this kernel is responsible for.
        // We will use float internally since it is more efficient on the GPU.
        // This is recommended for use cases where speed is more important than
        // accuracy, such as visualization.
        float sma = (float)orbit[idx].semi_major_axis;
        float ecc = (float)orbit[idx].eccentricity;
        float inc = (float)orbit[idx].inclination;
        float raan = (float)orbit[idx].raan;
        float aop = (float)orbit[idx].arg_of_perigee;
        float phi = (float)orbit[idx].mean_anomaly;

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
        float3 w = make_float3(cos(u) * cos(raan) - sin(u) * sin(raan) * cos(inc),
                               cos(u) * sin(raan) + sin(u) * cos(raan) * cos(inc),
                               sin(u) * sin(inc));

        float p = sma * (1.0f - powf(ecc,2.0f));
        float arg = 1.0f + (ecc * cos(true_anomaly));
        float r = p / EPSILON;
        if (arg > EPSILON) r = p / arg;

        // Write the position vector into the OPI::Population array.
        position[idx].x = (double)(w.x*r);
        position[idx].y = (double)(w.y*r);
        position[idx].z = (double)(w.z*r);

        // Finally, also write back the new mean anomaly into the orbit.
        orbit[idx].mean_anomaly = mean_anomaly;
    }
}

// Basic propagator that calculates cartesian position and unperturbed mean motion.
// This is the CUDA version. There are equivalent C++ and OpenCL plugins in the examples folder.
class BasicCUDA: public OPI::Propagator
{
    public:
        BasicCUDA(OPI::Host& host)
        {
            baseDay = 0;
        }

        virtual ~BasicCUDA()
        {

        }

        // This is the main function every plugin needs to implement to do the actual propagation.
        virtual OPI::ErrorCode runPropagation(OPI::Population& data, double julian_day, double dt )
        {
            // In this simple example, we don't have to fiddle with Julian dates. Instead, we'll just
            // look at the seconds that have elapsed since the first call of the propagator. The first
            // time runPropagation() is called, the given day is saved and then subtracted from the
            // following days. The remainder is converted to seconds and passed to the CUDA kernel.
            if (baseDay == 0) baseDay = julian_day;
            float seconds = (julian_day-baseDay)*86400.0 + dt;

            // Check whether there is a CUDA device we can use.
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);

            if(deviceCount >= 1)
            {
                OPI::Orbit* orbit = data.getOrbit(OPI::DEVICE_CUDA);
                OPI::Vector3* position = data.getPosition(OPI::DEVICE_CUDA);

                // Set kernel grid and block sizes based on the size of the population.
                // See CUDA manual for details on choosing proper block and grid sizes.
                int blockSize = 256;
                int gridSize = (data.getSize()+blockSize-1)/blockSize;

                // Call the CUDA kernel.
                kernel_propagate<<<gridSize, blockSize>>>(orbit, position, seconds, data.getSize());

                // The kernel writes to the Population's position and orbit vectors, so
                // these two have to be marked for updated values on the CUDA device.
                data.update(OPI::DATA_POSITION, OPI::DEVICE_CUDA);
                data.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA);

                return OPI::SUCCESS;
            }
            else return OPI::CUDA_REQUIRED;
        }

        // Especially with GPU-based propagators, you'll almost certainly also want to override
        // runIndexedPropagation() and runMultiTimePropagation(). The former propagates only
        // objects that appear in the given index list while the latter propgates objects to
        // individual Julian dates given in an array.
        // OPI provides basic implementations that call the (mandatory) runPropagation()
        // function in a loop but they are very inefficient and likely to severly impact the
        // performance of a CUDA- or OpenCL-based propagator.
        // I'll leave this to you to implement them properly. For runIndexedPropagation() it
        // is helpful to know that the IndexList synchronizes with the GPU just like the
        // Population - the functions IndexList::getData() and IndexList::update() work
        // just like their Population counterparts.
        OPI::ErrorCode runIndexedPropagation(OPI::Population& data, OPI::IndexList& indices, double julian_day, double dt)
        {
            return OPI::NOT_IMPLEMENTED;
        }

        OPI::ErrorCode runMultiTimePropagation(OPI::Population& data, double* julian_days, int length, double dt)
        {
            return OPI::NOT_IMPLEMENTED;
        }

        // Saving a member variable like baseDay in the propagator can lead to problems because
        // the host might change the propagation times or even the entire population without
        // notice. Therefore, plugin authors need to make sure that at least when disabling
        // and subsquently enabling the propagator, hosts can expect the propagator to
        // reset to its initial state.
        virtual OPI::ErrorCode runDisable()
        {
            baseDay = 0;
            return OPI::SUCCESS;
        }

        virtual OPI::ErrorCode runEnable()
        {
            return OPI::SUCCESS;
        }

        // The following functions need to be overridden to provide some information on
        // the plugin's capabilities.

        // Theoretically, the algorithm inside the CUDA kernel can handle backward propagation,
        // but the simplified handling of the input time cannot. Therefore, we'll return false
        // in this function.
        bool backwardPropagation()
        {
            return false;
        }

        // This propagator returns a position vector so we'll set this to true.
        // Defaults to false if not overridden.
        bool cartesianCoordinates()
        {
            return true;
        }

        // This propagator generates state vectors in an Earth-centered intertial
        // (ECI) reference frame. If not overridden, the default value is REF_NONE
        // if no cartesian coordinates are generated, REF_UNSPECIFIED otherwise.
        OPI::ReferenceFrame referenceFrame()
        {
            return OPI::REF_ECI;
        }

        // CUDA compute capability 3 is required for this plugin.
        int requiresCUDA()
        {
            return 3;
        }

        // This plugin does not require OpenCL so we return zero here.
        // This is also the default if not overridden.
        int requiresOpenCL()
        {
            return 0;
        }

        // This plugin is written for OPI version 1.0. (Default: 0)
        int minimumOPIVersionRequired()
        {
            return 1;
        }

    private:
        double baseDay;

};

#define OPI_IMPLEMENT_CPP_PROPAGATOR BasicCUDA

#include "OPI/opi_implement_plugin.h"
