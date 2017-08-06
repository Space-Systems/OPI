#include "OPI/opi_cpp.h"
#include <string>
#include <iostream>

// For this example, we'll use the new C++ wrapper, cl2.hpp.
// This requires some additional casting, see comments below.
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include "CL/cl2.hpp"

// Basic information about the plugin that can be queried by the host.
// OPI_PLUGIN_NAME is the most important as it serves as an identifier
// for the host to request a specific propagator. It should be descriptive
// and closely match the plugin's file name.
#define OPI_PLUGIN_NAME "BasicCL"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "Basic Mean Motion Converter - OpenCL version"

// Set the version number for the plugin here.
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0


// Basic propagator that calculates cartesian position and unperturbed mean motion.
// This is the OpenCL version. There are equivalent C++ and CUDA plugins in the examples folder.
// OpenCL code is more complex to set up than CUDA but you'll get a much wider variety of
// supported platforms, including multicore CPUs.
class BasicCL: public OPI::Propagator
{
	public:
        BasicCL(OPI::Host& host)
		{
			initialized = false;
            baseDay = 0;

            // Get GPU support module from host.
            // The OpenCL version of the support module will provide important additional
            // information compared to the CUDA version, such as the OpenCL context and
            // default command queue which are required to run kernels.
            clSupport = host.getGPUSupport();
		}

        virtual ~BasicCL()
		{

		}

        // Auxiliary function to create the OpenCL kernel containing the propagation code.
        // It is executed the first time runPropagation() is called.
        cl::Kernel createPropagator()
		{
			// cl_int for OpenCL error reporting
			cl_int err;

            // Create the kernel program. OPI's OpenCL support module returns the context
            // and device pointers as C types - to write a propagator using the OpenCL C++
            // API, these need to be wrapped into their respective C++ objects.
            // retainOwnership needs to be set to true to let OPI manage the context.
            cl::Context context = cl::Context(*clSupport->getOpenCLContext(), true);
            cl::Program program = cl::Program(context, kernelCode, false, &err);
            if (err != CL_SUCCESS) std::cout << "Error creating program: " << err << std::endl;

            // Build the kernel for the default device. Again, the C type from OPI's OpenCL
            // module needs to be wrapped into the cl::Device class. Again, make sure
            // retainOwnership is true to let OPI manage the device.
            cl::Device device = cl::Device(*clSupport->getOpenCLDevice(), true);
            err = program.build({device});
            if (err != CL_SUCCESS) std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

			// create the kernel object
            cl::Kernel kernel = cl::Kernel(program, "cl_propagate", &err);
			if (err != CL_SUCCESS) {
                std::cout << "Error creating kernel: " << err << std::endl;
			}
            else std::cout << "Kernel created." << std::endl;
			return kernel;
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

            // Create the OpenCL kernel on the first run.
            if (!initialized) {
                propagator = createPropagator();
                initialized = true;
            }

            // cl_int for OpenCL error reporting.
            cl_int err;

            // Calling getOrbit, getObjectProperties, etc. with the DEVICE_CUDA flag will return
            // cl_mem instances in the OpenCL implementation. They must be explicitly cast to
            // cl_mem before they can be used as kernel arguments. This step will also trigger
            // the memory transfer from host to OpenCL device if required.
            cl_mem orbit = reinterpret_cast<cl_mem>(data.getOrbit(OPI::DEVICE_CUDA));
            cl_mem position = reinterpret_cast<cl_mem>(data.getPosition(OPI::DEVICE_CUDA));

            // To use the OpenCL C++ API, we also need to create a cl::Buffer object from
            // the cl_mem instance. Again, the retainObject flag of the cl::Buffer constructor
            // must be set to true, otherwise OPI will lose ownership of the memory pointer
            // which will cause subsequent copy operations to fail.
            cl::Buffer orbitBuffer = cl::Buffer(orbit, true);
            cl::Buffer positionBuffer = cl::Buffer(position, true);

            // The cl::Buffer objects can then be used as kernel arguments.
            err = propagator.setArg(0, orbitBuffer);
            if (err != CL_SUCCESS) std::cout << "Error setting orbit data: " << err << std::endl;
            err = propagator.setArg(1, positionBuffer);
            if (err != CL_SUCCESS) std::cout << "Error setting position data: " << err << std::endl;

            // set remaining arguments (julian_day and dt)
            propagator.setArg(2, seconds);
            propagator.setArg(3, data.getSize());

            // Get the command queue (retainOwnership = true, you get the idea...)
            cl::CommandQueue queue = cl::CommandQueue(*clSupport->getOpenCLQueue(), true);
            // Enqueue the kernel with a 1-dimensional NDRange matching the size of the population.
            err = queue.enqueueNDRangeKernel(propagator, cl::NullRange, cl::NDRange(data.getSize()), cl::NullRange);
            if (err != CL_SUCCESS) std::cout << "Error running kernel: " << err << std::endl;

            // Wait for the kernel to finish.
            queue.finish();

            // The kernel writes to the Population's position and orbit vectors, so
            // these two have to be marked for updated values on the OpenCL ("CUDA") device.
            data.update(OPI::DATA_CARTESIAN, OPI::DEVICE_CUDA);
            data.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA);

            return OPI::SUCCESS;
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
            initialized = false;
            baseDay = 0;
            return OPI::SUCCESS;
        }

        virtual OPI::ErrorCode runEnable()
        {
            return OPI::SUCCESS;
        }

        // The following functions need to be overridden to provide some information on
        // the plugin's capabilities.

        // Theoretically, the algorithm can handle backward propagation,
        // but the simplified handling of the input time cannot. Therefore, we'll
        // return false in this function. Also defaults to false if not overridden.
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

        // This plugin does not require CUDA so we return zero here.
        // This is also the default if not overridden.
        int requiresCUDA()
        {
            return 0;
        }

        // This plugin requires OpenCL version 1.0 or greater.
        int requiresOpenCL()
        {
            return 1;
        }

        // This plugin is written for OPI version 1.0. (Default: 0)
        int minimumOPIVersionRequired()
        {
            return 1;
        }

	private:
        cl::Kernel propagator;
        OPI::GpuSupport* clSupport;
		bool initialized;
        double baseDay;

        // Define kernel source code as a string literal (requires C++11).
        std::string kernelCode = R"(

            // The orbit and vector types need to be redefined in OpenCL.
            // Make sure they are identical to their counterparts on the host!
            struct Orbit {
                double semi_major_axis;
                double eccentricity;
                double inclination;
                double raan;
                double arg_of_perigee;
                double mean_anomaly;
                double bol;
                double eol;
            };

            struct Vector3 {
                double x;
                double y;
                double z;
            };

            // Auxiliary OpenCL function to convert mean anomaly to eccentric anomaly.
            // Identical to the function in the C++ example.
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

            // OpenCL kernel that does the actual transformations. Equivalent to the basic CUDA
            // and CPP examples with only minor changes.
            kernel void cl_propagate(global struct Orbit* orbit, global struct Vector3* position, float seconds, int size)
            {
                // Get ID for this kernel...
                int i = get_global_id(0);
                // ...and make sure it doesn't exceed the Population size.
                // Since we gave OpenCL the exact number of kernels to execute, this should not happen.
                if (i < size)
                {
                    // Store orbit data from the object this kernel is responsible for.
                    // We will use float internally since it is more efficient on the GPU.
                    // This is recommended for use cases where speed is more important than
                    // accuracy, such as visualization.
                    float sma = (float)orbit[i].semi_major_axis;
                    float ecc = (float)orbit[i].eccentricity;
                    float inc = (float)orbit[i].inclination;
                    float raan = (float)orbit[i].raan;
                    float aop = (float)orbit[i].arg_of_perigee;
                    float phi = (float)orbit[i].mean_anomaly;

                    // Define some auxiliary constants.
                    float PI = 3.1415926f;
                    float RMUE = 398600.5f;
                    float EPSILON = 0.00001f;

                    // Confine the input time to the object's orbit period.
                    float orbit_period = 2.0f * PI * sqrt(pow(sma,3.0f) / RMUE);
                    float t = fmod(seconds, orbit_period);

                    // Calculate the mean anomaly and eccentric anomaly.
                    // Note: This disregards the initial mean anomaly given in the Population -
                    // avoid this in production plugins.
                    // Use fmod, pow and sqrt in OpenCL instead of fmodf, powf and sqrtf.
                    float mean_anomaly = fmod(sqrt((RMUE * t * t) / pow(sma,3.0f)), 2.0f*PI);
                    float excentric_anomaly = mean2eccentric(mean_anomaly, ecc);

                    // Convert eccentric anomaly to true anomaly.
                    float sin_ea = sin(excentric_anomaly/2.0f);
                    float cos_ea = cos(excentric_anomaly/2.0f);
                    float true_anomaly = 2.0f * atan(sqrt((1.0f + ecc)/(1.0f - ecc)) * sin_ea/cos_ea);

                    // Based on the true anomaly, calculate cartesian object coordinates.
                    float u = true_anomaly + aop;
                    // This variable needs to be declared with the "struct" identifier in OpenCL.
                    struct Vector3 w;
                    w.x = cos(u) * cos(raan) - sin(u) * sin(raan) * cos(inc);
                    w.y = cos(u) * sin(raan) + sin(u) * cos(raan) * cos(inc);
                    w.z = sin(u) * sin(inc);

                    float p = sma * (1.0f - pow(ecc,2.0f));
                    float arg = 1.0f + (ecc * cos(true_anomaly));
                    float r = p / EPSILON;
                    if (arg > EPSILON) r = p / arg;

                    // Write the position vector into the OPI::Population array.
                    position[i].x = (double)(w.x*r);
                    position[i].y = (double)(w.y*r);
                    position[i].z = (double)(w.z*r);

                    // Finally, also write back the new mean anomaly into the orbit.
                    orbit[i].mean_anomaly = (double)mean_anomaly;
                }
            }

        )";
};

#define OPI_IMPLEMENT_CPP_PROPAGATOR BasicCL

#include "OPI/opi_implement_plugin.h"

