#include "OPI/opi_cpp.h"
#include <iostream>

// Basic information about the plugin that can be queried by the host.
// OPI_PLUGIN_NAME is the most important as it serves as an identifier
// for the host to request a specific propagator. It should be descriptive
// and closely match the plugin's file name.
#define OPI_PLUGIN_NAME "PropertiesExampleCPP"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "Example propagator showing how to use PropagatorProperties"

// Set the version number for the plugin here.
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0


// This propagator does not propagate at all. Its purpose is to demonstrate the use
// of PropagatorProperties. PPs are variables stored inside a propagator that can be
// queried and set by the host or via a config file. Their purpose is to communicate
// additional settings to the propagator that are not covered by the generic interface.
// It is recommended that plugin authors supply a commented config file explaining the
// PPs and their default values. Host authors should always query the plugin's PPs and
// display them in the user interface. An example config file is provided for this
// plugin.
class PropertiesCPP: public OPI::Propagator
{
    private:

        // These are member variables that will store our PropagatorProperties.
        int testproperty_int;
        float testproperty_float;
        double testproperty_double;
        int testproperty_int_array[5];
        float testproperty_float_array[2];
        double testproperty_double_array[4];
        std::string testproperty_string;

        // Function to (re)set properties to their default values.
        // The propagator will fall back to these if the host does not set
        // any properties, and no config file is found.
        void setDefaultPropertyValues()
        {
            testproperty_int = 0;
            testproperty_float = 0.0f;
            testproperty_double = 42.0;
            testproperty_string = "test";
            for(int i = 0; i < 5; ++i)
            {
                testproperty_int_array[i] = i;
            }
            testproperty_float_array[0] = 4.0f;
            testproperty_float_array[1] = 2.0f;
            for(int i = 0; i < 4; ++i)
            {
                testproperty_double_array[i] = 9.0 * i;
            }
        }

	public:

        // For C++ propagators, the constructor should be used to register PPs and
        // set default values. When a plugin is loaded, the host will attempt to
        // load a config file AFTER the constructor was called. This means that the config
        // file takes priority over member variable defaults, and property settings by the
        // host take priority over the config file. The name of the config file is the file
        // name of the plugin (not the plugin name defined above!), with the .dll/.so/.dynlib
        // suffix replaced with .cfg. It should reside in the same directory as the plugin.
        PropertiesCPP(OPI::Host& host)
		{
            // Call the above function to set internal variables for PPs
            setDefaultPropertyValues();

            // Define available properties and link them to the internal variables.
            registerProperty("ThisIsAnInteger", &testproperty_int);
            registerProperty("ThisIsAFloat", &testproperty_float);
            registerProperty("ThisIsADouble", &testproperty_double);
            registerProperty("ThisIsAString", &testproperty_string);
            registerProperty("ThisIsAnIntegerArray", testproperty_int_array, 5);
            registerProperty("ThisIsAFloatArray", testproperty_float_array, 2);
            registerProperty("ThisIsADoubleArray", testproperty_double_array, 4);
		}

        virtual ~PropertiesCPP()
		{

		}

        // The actual propagation function does nothing, except print some of the
        // property values. To see how actual propagation works, check the "basic"
        // examples in this folder.
        virtual OPI::ErrorCode runPropagation(OPI::Population& data, double julian_day, double dt )
		{
			std::cout << "Test int: " << testproperty_int << std::endl;
			std::cout << "Test float: " <<  testproperty_float << std::endl;
			std::cout << "Test string: " << testproperty_string << std::endl;
			return OPI::SUCCESS;
		}

        // When disabling and enabling the propagator, the host will expect it to reset
        // to its initial state, so we will reset the PropagatorProperties
        // to the default values. On disabling, call loadConfigFile() to try reading
        // the default configuration file that was specified when the propagator was
        // first loaded. You can also call loadConfigFile(std::string filename) to
        // load a configuration from a different file.
        // Note that runEnable() may be called after the host set its own values for
        // the PropagatorProperties so if you have to reset them, do it in runDisable()!
        virtual OPI::ErrorCode runDisable()
        {
            setDefaultPropertyValues();
            loadConfigFile();
            return OPI::SUCCESS;
        }

        virtual OPI::ErrorCode runEnable()
        {
            return OPI::SUCCESS;
        }

        // Since this propagator does nothing, we can return anything here.
        bool backwardPropagation()
        {
            return true;
        }

        // This propagator does not generate a position vector.
        bool cartesianCoordinates()
        {
            return false;
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
};

#define OPI_IMPLEMENT_CPP_PROPAGATOR PropertiesCPP

#include "OPI/opi_implement_plugin.h"

