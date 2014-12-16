# Example OPI host written in Python.
# As with the other examples, libOPI.so needs to be in your library path
# in order to run this script. On Linux, if you didn't install OPI into
# a system path, you can use the LD_LIBRARY_PATH variable, e.g.
# LD_LIBRARY_PATH=</path/to/opi/installation>/lib python host_python_example.py

# Import bindings from opi.py (created with the included shell script).
from opi import *

# Create host object
host = OPI_createHost()

# Load plugins from given directory
OPI_loadPlugins(host, "plugins")

# Create population data with 200 objects (must be done after loading the plugins)
data = OPI_createData(host, 200)

# Fetch a pointer to the orbit data of the population we just created
orbits = OPI_Population_getOrbit(data)

# Fetch a pointer to the object properties array of the population
props = OPI_Population_getObjectProperties(data)

# Set initial values for objects
# Don't forget setting BOL and EOL to zero if you don't use them!
# Also remember to set sensible values for the other properties
# as leaving them uninitialized can lead to strange propagation
# results. This is especially true for area to mass ratio, drag
# coefficient and reflectivity in combination with atmospheric
# perturbation models.
for i in range(0, OPI_Population_getSize(data)):
	orbits[i].semi_major_axis = 6800
	orbits[i].eccentricity = 0.001
	orbits[i].inclination = 23.5
	orbits[i].raan = 0
	orbits[i].arg_of_perigee = 0
	orbits[i].mean_anomaly = 0
	orbits[i].bol = 0
	orbits[i].eol = 0
	props[i].id = i
	props[i].mass = 100
	props[i].diameter = 10
	props[i].area_to_mass = 1
	props[i].drag_coefficient = 2.2
	props[i].reflectivity = 1.3

# Notify OPI of the updated values
OPI_Population_update(data, OPI_DEVICE_HOST)

# Print information about available propagators
print "Propagators:"
for i in range(0, OPI_getPropagatorCount(host)):
	name = OPI_Module_getName(OPI_getPropagatorByIndex(host, i))
	author = OPI_Module_getAuthor(OPI_getPropagatorByIndex(host, i))
	print "#",i,": ",name," by ",author

# Select a propagator by name
propagator = OPI_getPropagatorByName(host, "Fortran Example Propagator")

# Start propagation if a valid propagator was selected
if propagator:
	# Propagate all objects with the chosen propagator and the given time stamp and step size
	OPI_Propagator_propagateAll(propagator, data, 2450980.0, 86400.0)
	# Update orbit pointer 
	orbits = OPI_Population_getOrbit(data)
	# Print (partial) propagation results for one object
	print(str(orbits[1].semi_major_axis) + " " + str(orbits[1].eccentricity) + " " + str(orbits[1].mean_anomaly))
else:
	print("Propagator not found!")

# Clean up
OPI_destroyData(data)
OPI_destroyHost(host)
