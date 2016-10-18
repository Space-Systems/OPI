#include <cstdlib>
#include <iostream>
#include "OPI/opi_cpp.h"
#include <cassert>
// just testing the error callback funcionality
void ErrorCallback(OPI_Host c_host, int code, void* privateData)
{
	// translate host to cpp host object
	//OPI::Host* host = (OPI::Host*)(c_host);
	std::cout << "OPI encountered an error: " << OPI::ErrorMessage(code) << std::endl;
	assert(0);
	exit(1);
}

int main(int argc, char* argv[])
{
	// initialize host
	OPI::Host host;

	host.setErrorCallback(&ErrorCallback, 0);
	// load plugins
	host.loadPlugins("plugins",OPI::Host::PLATFORM_OPENCL);

	// create Data object AFTER! loading all plugins
	OPI::Population data(host, 200);

	// list each loaded propagator
	for(int i = 0; i < host.getPropagatorCount();++i)
	{
		OPI::Propagator* propagator = host.getPropagator(i);
		std::cout << "Propagator #" << i << ": " << propagator->getName() << std::endl
							<< "  " << propagator->getDescription() << std::endl
							<< "  by: " << propagator->getAuthor() << std::endl;
	}

	// fetch a specific propagator
    OPI::Propagator* propagator = host.getPropagator("OpenCL CPP Example Propagator");

	if (propagator) {

		std::cout << "Propagator properties:" << std::endl;
		for(int i = 0; i < propagator->getPropertyCount(); ++i)
		{
			std::string name = propagator->getPropertyName(i);
			std::cout << name;
			switch(propagator->getPropertyType(i))
			{
				case OPI::TYPE_UNKNOWN:
					break;
				case OPI::TYPE_INTEGER:
					std::cout << "(TYPE_INTEGER) value: " << propagator->getPropertyInt(i);
					break;
				case OPI::TYPE_FLOAT:
					std::cout << "(TYPE_FLOAT) value: " << propagator->getPropertyFloat(i);
					break;
				case OPI::TYPE_DOUBLE:
					std::cout << "(TYPE_DOUBLE) value: " << propagator->getPropertyDouble(i);
					break;
				case OPI::TYPE_STRING:
					std::cout << "(TYPE_STRING) value: " << propagator->getPropertyString(i);
					break;
				case OPI::TYPE_INTEGER_ARRAY:
					std::cout << "(TYPE_INTEGER_ARRAY) values: " << propagator->getPropertyInt(i, 0);
					for(int j = 1; j < propagator->getPropertySize(i); ++j)
						std::cout << ", " << propagator->getPropertyInt(i, j);
					break;
				case OPI::TYPE_FLOAT_ARRAY:
					std::cout << "(TYPE_FLOAT_ARRAY) values: " << propagator->getPropertyFloat(i, 0);
					for(int j = 1; j < propagator->getPropertySize(i); ++j)
						std::cout << ", " << propagator->getPropertyFloat(i, j);
					break;
				case OPI::TYPE_DOUBLE_ARRAY:
					std::cout << "(TYPE_DOUBLE_ARRAY) values: " << propagator->getPropertyDouble(i, 0);
					for(int j = 1; j < propagator->getPropertySize(i); ++j)
						std::cout << ", " << propagator->getPropertyDouble(i, j);
					break;
			}
			std::cout << std::endl;
		}

		std::cout << "Using propagator: " << propagator->getName() << std::endl;
		// request data pointer for orbital data
		OPI::Orbit* orbit = data.getOrbit();
		// initialize some values
		for(int i = 0; i < data.getSize(); ++i)
		{
			orbit[i].arg_of_perigee = i;
			orbit[i].semi_major_axis = 6800.0f;
		}
		// inform the data about changes inside our orbit structure
		data.update(OPI::DATA_ORBIT);

		// run a propagation
		propagator->propagate(data, 0, 0);

		// remove some objects
		OPI::IndexList list(host);
		list.add(32);
		list.add(54);
		data.remove(list);

		// refresh data pointer for orbital data again
		// this will automatically sync the data between different devices
		orbit = data.getOrbit(OPI::DEVICE_HOST);

		for(int i = 0; i < 10; ++i)
		{
			std::cout << orbit[i].inclination << " " << orbit[i].arg_of_perigee << " " << orbit[i].semi_major_axis << std::endl;
		}
	}
	else
	{
		std::cout << "Propagator not found" << std::endl;
	}
	return EXIT_SUCCESS;
}
