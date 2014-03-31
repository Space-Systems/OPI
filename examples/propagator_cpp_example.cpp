#include "OPI/opi_cpp.h"

// some plugin information
#define OPI_PLUGIN_NAME "CPP Example Propagator"
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
		}

		virtual ~TestPropagator()
		{

		}

		virtual OPI::ErrorCode runPropagation(OPI::ObjectData& data, double julian_day, float dt )
		{
			std::cout << "Test int: " << testproperty_int << std::endl;
			std::cout << "Test float: " <<  testproperty_float << std::endl;
			std::cout << "Test string: " << testproperty_string << std::endl;
			OPI::Orbit* orbit = data.getOrbit();
			for(int i = 0; i < data.getSize(); ++i)
			{
				orbit[i].inclination = i;
			}
			data.update(OPI::DATA_ORBIT);
			return OPI::NO_ERROR;
		}
	private:
		int testproperty_int;
		float testproperty_float;
		double testproperty_double;
		int testproperty_int_array[5];
		float testproperty_float_array[2];
		double testproperty_double_array[4];
		std::string testproperty_string;
};

#define OPI_IMPLEMENT_CPP_PROPAGATOR TestPropagator

#include "OPI/opi_implement_plugin.h"

