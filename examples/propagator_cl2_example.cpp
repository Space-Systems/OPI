#define CL_TARGET_OPENCL_VERSION 120
#include "OPI/opi_cpp.h"

// For this example, we'll use the new C++ wrapper, cl2.hpp.
// This requires some additional casting, see comments below.
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include "CL/cl2.hpp"

#include <string>
#include <iostream>

// some plugin information
#define OPI_PLUGIN_NAME "OpenCL CPP Example Propagator"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "A simple test"
// the plugin version
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0

class MyCL2Propagator: public OPI::Propagator
{
public:
    MyCL2Propagator(OPI::Host& host)
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
        // The OpenCL version will provide important additional information such as the
        // OpenCL context and default command queue.
        clSupport = host.getGPUSupport();
    }

    virtual ~MyCL2Propagator()
    {

    }

    cl::Kernel createPropagator()
    {
        // cl_int for OpenCL error reporting
        cl_int err;

        // Define kernel source code as a string literal (requires C++11).
        std::string kernelCode = R"(
                                 struct Orbit {float semi_major_axis;float eccentricity;float inclination;float raan;float arg_of_perigee;float mean_anomaly;};
                                 typedef struct Aux {double a; char c; int i; double b;};
                                 void test(global struct Aux* b) { b->i = b->i + 1; }
                                 __kernel void propagate(__global struct Orbit* orbit, __global char* bytes, double julian_day, double dt) {
                                 int i = get_global_id(0);
                                 __global struct Aux* b = (__global struct Aux*)&bytes[i*sizeof(__global struct Aux)];
                                 test(b);
                                 orbit[i].semi_major_axis = i;
                                 })";

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
        cl::Kernel kernel = cl::Kernel(program, "propagate", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Error creating kernel: " << err << std::endl;
        }
        else std::cout << "Kernel created." << std::endl;
        return kernel;
    }

    virtual OPI::ErrorCode runPropagation(OPI::Population& population, double julian_day, double dt, OPI::PropagationMode mode, OPI::IndexList* indices)
    {
        if (mode == OPI::MODE_INDIVIDUAL_EPOCHS)
        {
            // If updating from OPI 2015, move code from runMultiTimePropagation() here instead.
            return OPI::NOT_IMPLEMENTED;
        }

        if (indices != nullptr)
        {
            // If updating from OPI 2015, move code from runIndexedPropagation() here instead.
            return OPI::NOT_IMPLEMENTED;
        }

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

        std::cout << population.getSize() << std::endl;

        // Calling getOrbit and getObjectProperties with the DEVICE_CUDA flag will return
        // cl_mem instances in the OpenCL implementation. They must be explicitly cast to
        // cl_mem before they can be used as kernel arguments. This step will also trigger
        // the memory transfer from host to OpenCL device.
        cl_mem orbit = reinterpret_cast<cl_mem>(population.getOrbit(OPI::DEVICE_CUDA));
        cl_mem bytes = reinterpret_cast<cl_mem>(population.getBytes(OPI::DEVICE_CUDA));

        // To use the OpenCL C++ API, we also need to create a cl::Buffer object from
        // the cl_mem instance. The retainObject flag of the cl::Buffer constructor
        // must be set to true, otherwise OPI will lose ownership of the memory pointer
        // which will cause subsequent copy operations to fail.
        cl::Buffer orbitBuffer = cl::Buffer(orbit, true);
        cl::Buffer bytesBuffer = cl::Buffer(bytes, true);

        // The cl::Buffer object can then be used as a kernel argument.
        err = propagator.setArg(0, orbitBuffer);
        err = propagator.setArg(1, bytesBuffer);
        if (err != CL_SUCCESS) std::cout << "Error setting population data: " << err << std::endl;

        // set remaining arguments (julian_day and dt)
        propagator.setArg(2, julian_day);
        propagator.setArg(3, dt);

        // Enqueue the kernel.
        const size_t problemSize = population.getSize();
        cl::CommandQueue queue = cl::CommandQueue(*clSupport->getOpenCLQueue(), true);
        err = queue.enqueueNDRangeKernel(propagator, cl::NullRange, cl::NDRange(problemSize), cl::NullRange);
        if (err != CL_SUCCESS) std::cout << "Error running kernel: " << err << std::endl;

        // wait for the kernel to finish
        queue.finish();

        // Don't forget to notify OPI of the updated data on the device!
        population.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA);
        population.update(OPI::DATA_BYTES, OPI::DEVICE_CUDA);

        return OPI::SUCCESS;
    }

    int requiresOpenCL()
    {
        return 1;
    }

    int requiresCUDA()
    {
        return 0;
    }

    // This plugin is written for OPI version 2019.8
    int minimumOPIVersionRequired()
    {
        return 2019;
    }

    int minorOPIVersionRequired()
    {
        return 8;
    }

    OPI::ReferenceFrame referenceFrame()
    {
        return OPI::REF_UNSPECIFIED;
    }

private:
    int testproperty_int;
    float testproperty_float;
    double testproperty_double;
    int testproperty_int_array[5];
    float testproperty_float_array[2];
    double testproperty_double_array[4];
    std::string testproperty_string;
    cl::Kernel propagator;
    OPI::GpuSupport* clSupport;
    bool initialized;
};

#define OPI_IMPLEMENT_CPP_PROPAGATOR MyCL2Propagator

#include "OPI/opi_implement_plugin.h"

