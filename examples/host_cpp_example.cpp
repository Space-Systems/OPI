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
	host.loadPlugins("plugins");

	// create Data object AFTER! loading all plugins
	OPI::ObjectData data(host, 200);

	// list each loaded propagator
	for(int i = 0; i < host.getPropagatorCount();++i)
	{
		OPI::Propagator* propagator = host.getPropagator(i);
		std::cout << "Propagator #" << i << ": " << propagator->getName() << std::endl
							<< "  " << propagator->getDescription() << std::endl
							<< "  by: " << propagator->getAuthor() << std::endl;
	}

	// fetch a specific propagator
	OPI::Propagator* propagator = host.getPropagator("CPP Example Propagator");
	propagator->setProperty("int",13);
	propagator->setProperty("float", 3.141f);
	propagator->setProperty("string", "abcde");

	std::cout << "Propagator properties:" << std::endl;
	for(int i = 0; i < propagator->getPropertyCount(); ++i)
	{
		std::string name = propagator->getPropertyName(i);
		std::cout << name << ": " << propagator->getPropertyString(name) << std::endl;
	}
	if(propagator)
	{
		std::cout << "Using propagator: " << propagator->getName() << std::endl;
		// request data pointer for orbital data
		OPI::Orbit* orbit = data.getOrbit();
		// initialize some values
		for(int i = 0; i < data.getSize(); ++i)
		{
			orbit[i].arg_of_perigee = i;
		}
		// inform the data about changes inside our orbit structure
		data.update(OPI::DATA_ORBIT);

		// run a propagation
		propagator->propagate(data, 0, 0, 0 );

		// remove some objects
		OPI::IndexList list(host);
		list.add(32);
		list.add(54);
		data.remove(list);

		// refresh data pointer for orbital data again
		// this will automatically sync the data between different devices
		orbit = data.getOrbit();
		for(int i = 0; i < data.getSize(); ++i)
		{
			std::cout << orbit[i].inclination
								<< " "
								<< orbit[i].arg_of_perigee
								<< " "
								<< orbit[i].semi_major_axis
								<< std::endl;
		}
	}
	else
	{
		std::cout << "Propagator not found" << std::endl;
	}
	return EXIT_SUCCESS;
}
