#include "OPI/opi_cpp.h"
#include "CL/cl.h"

#include <typeinfo> //testing

// some plugin information
#define OPI_PLUGIN_NAME "OpenCL C Example Propagator"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "A simple test"
// the plugin version
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0
#include <iostream>
class TestPropagator:
		public OPI::Propagator
{
	public:
		TestPropagator(OPI::Host& host)
		{
			testproperty_int = 0;
			testproperty_float = 0.0f;
			testproperty_double = 42.0;
			testproperty_string = "test";
			for(int i = 0; i < 5; ++i)
				testproperty_int_array[i] = i;
			testproperty_float_array[0] = 4.0f;
			testproperty_float_array[1] = 2.0f;
			for(int i = 0; i < 4; ++i)
				testproperty_double_array[i] = 9.0 * i;
			registerProperty("int", &testproperty_int);
			registerProperty("float", &testproperty_float);
			registerProperty("double", &testproperty_double);
			registerProperty("string", &testproperty_string);
			registerProperty("int_array", testproperty_int_array, 5);
			registerProperty("float_array", testproperty_float_array, 2);
			registerProperty("double_array", testproperty_double_array, 4);

			initialized = false;

            // Get GPU support module from host.
            clSupport = host.getGPUSupport();
		}

		virtual ~TestPropagator()
		{

		}

		cl_kernel createPropagator()
		{
			// cl_int for OpenCL error reporting
			cl_int err;

			// define kernel source code
			const char* kernelCode = "\n" \
				"struct Orbit {float semi_major_axis;float eccentricity;float inclination;float raan;float arg_of_perigee;float mean_anomaly;float bol;float eol;}; \n" \
                "__kernel void propagate(__global struct Orbit* orbit, double julian_day, double dt) { \n" \
				"int i = get_global_id(0); \n" \
				"orbit[i].semi_major_axis += i + julian_day + dt; \n" \
				"} \n";

			// create kernel program
            cl_program program = clCreateProgramWithSource(*clSupport->getOpenCLContext(), 1, (const char**)&kernelCode, NULL, &err);
			if (err != CL_SUCCESS) std::cout << "Error creating program: " << err << std::endl;

			// build kernel for default device
            const cl_device_id device = *clSupport->getOpenCLDevice();
			err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

			// print build log for debugging purposes
			char buildLog[2048];
            clGetProgramBuildInfo(program, *clSupport->getOpenCLDevice(), CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            std::cout << "--- Build log ---\n " << buildLog << std::endl;

			// create the kernel object
			cl_kernel kernel = clCreateKernel(program, "propagate", &err);
			if (err != CL_SUCCESS) {
				std::cout << "Error creating kernel: " << err << std::endl;
			}
            std::cout << "Kernel created." << std::endl;
			return kernel;
		}

        virtual OPI::ErrorCode runPropagation(OPI::Population& population, double julian_day, double dt )
		{
			std::cout << "Test int: " << testproperty_int << std::endl;
			std::cout << "Test float: " <<  testproperty_float << std::endl;
			std::cout << "Test string: " << testproperty_string << std::endl;

			if (!initialized) {
				// Create propagator kernel from embedded source code
				propagator = createPropagator();
				initialized = true;
			}

			// cl_int for OpenCL error reporting
			cl_int err;

			// Check whether kernel was successfully built
			if (propagator) {

				// Calling getOrbit and getObjectProperties with the DEVICE_CUDA flag will return
				// cl_mem objects in the OpenCL implementation. They must be explicitly cast to
				// cl_mem before they can be used as kernel arguments. This step will also trigger
				// the memory transfer from host to OpenCL device.
                std::cout << population.getSize() << std::endl;
				cl_mem orbit = reinterpret_cast<cl_mem>(population.getOrbit(OPI::DEVICE_CUDA));
				err = clSetKernelArg(propagator, 0, sizeof(cl_mem), &orbit);
                if (err != CL_SUCCESS) std::cout << "Error setting population data: " << err << std::endl;

				// set remaining arguments (julian_day and dt)
				err = clSetKernelArg(propagator, 1, sizeof(double), &julian_day);
                if (err != CL_SUCCESS) std::cout << "Error setting jd data: " << err << std::endl;
				err = clSetKernelArg(propagator, 2, sizeof(float), &dt);
                if (err != CL_SUCCESS) std::cout << "Error setting dt data: " << err << std::endl;

				// run the kernel
				const size_t s = population.getSize();
                err = clEnqueueNDRangeKernel(*clSupport->getOpenCLQueue(), propagator, 1, NULL, &s, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS) std::cout << "Error running kernel: " << err << std::endl;

				// wait for the kernel to finish
                clFinish(*clSupport->getOpenCLQueue());

				// Don't forget to notify OPI of the updated data on the device!
				population.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA);

				return OPI::SUCCESS;
			}
			return OPI::INVALID_DEVICE;
		}

		int requiresOpenCL()
		{
			return 1;
		}

		int requiresCUDA()
		{
			return 0;
		}

        int minimumOPIVersionRequired()
        {
            return 1;
        }

	private:
		int testproperty_int;
		float testproperty_float;
		double testproperty_double;
		int testproperty_int_array[5];
		float testproperty_float_array[2];
		double testproperty_double_array[4];
		std::string testproperty_string;
		cl_kernel propagator;
        OPI::GpuSupport* clSupport;
		bool initialized;
};

#define OPI_IMPLEMENT_CPP_PROPAGATOR TestPropagator

#include "OPI/opi_implement_plugin.h"

