#include <stdlib.h>
#include <stdio.h>
#include "OPI/opi_c_bindings.h"

int main(int argc, char* argv[])
{
	OPI_Host host;
	OPI_ObjectData data;
	OPI_Propagator propagator;
	OPI_Orbit* orbitals;
	int i;

	// create our host object
	host = OPI_createHost();

	// initialize plugin_dirs
	OPI_loadPlugins(host, "plugins");

	// create our data object AFTER! initializing
	// and loading the plugins
	data = OPI_createData(host, 200);

	printf("Registered Propagator:\n");
	for(i = 0; i < OPI_getPropagatorCount(host); ++i)
	{
		printf("#%i: %s by %s\n", i, OPI_Module_getName(OPI_getPropagatorByIndex(host, i)),
					 OPI_Module_getAuthor(OPI_getPropagatorByIndex(host, i)));
	}

	// retrieve a specific propagator
	propagator = OPI_getPropagatorByName(host, "Fortran Example Propagator");

	if(propagator)
	{
		// run normal propagation
		OPI_Propagator_propagateAll(propagator, data, 0, 0, 0);

		// retrieve orbital data
		orbitals = OPI_ObjectData_getOrbit(data);

		// just print it out here
		for(i = 0; i < 200; ++i)
			printf("%f\n", orbitals[i].inclination);

	}
	else {
		printf("Propagator not found!");
	}

	// destroy our data object
	OPI_destroyData(data);

	// destroy our host object
	OPI_destroyHost(host);

	return EXIT_SUCCESS;
}
