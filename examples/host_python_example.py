# Example OPI host written in Python.
# As with the other examples, libOPI.so needs to be in your library path
# in order to run this script. On Linux, if you didn't install OPI into
# a system path, you can use the LD_LIBRARY_PATH variable, e.g.
# LD_LIBRARY_PATH=</path/to/opi/installation>/lib python host_python_example.py

# Import bindings from OPI.py (needs to be in your PYTHONPATH)
import OPI

# Create host object
host = OPI.Host()

# Load plugins from given directory
host.loadPlugins("plugins", OPI.Host.PLATFORM_NONE)

# Print information about available propagators
print("Propagators:")
for i in range(0, host.getPropagatorCount()):
    name = host.getPropagator(i).getName()
    author = host.getPropagator(i).getAuthor()
    print("  #",i,":",name,"by",author)

# Create a population with one object
population = OPI.Population(host,1)

# Set object's orbit
orbit = OPI.Orbit()
orbit.semi_major_axis = 7150.59176
orbit.eccentricity = 0.0012347
orbit.inclination = 1.022
orbit.raan = 0.3
orbit.arg_of_perigee = 0.66
orbit.mean_anomaly = 0.1
population.setOrbit(0, orbit)

# If the propagator needs state vectors, use this instead:
#position = OPI.Vector3(615.119526, -7095.644839, -678.668352)
#velocity = OPI.Vector3(0.390367, 0.741902, -7.396980)
#population.setPosition(0, position)
#population.setVelocity(0, velocity)

# Set object's properties
props = OPI.ObjectProperties()
props.mass = 1.3
props.area_to_mass = 1.0
props.diameter = 0.01
props.drag_coefficient = 2.2
props.reflectivity = 1.3
population.setObjectProperties(0, props)

# Tell OPI about the population updates
#population.update(OPI.DATA_POSITION, OPI.DEVICE_HOST)
#population.update(OPI.DATA_VELOCITY, OPI.DEVICE_HOST)
population.update(OPI.DATA_ORBIT, OPI.DEVICE_HOST)
population.update(OPI.DATA_PROPERTIES, OPI.DEVICE_HOST)

# Select a propagator by name
propagator = host.getPropagator("BasicCPP")

# Start propagation if a valid propagator was selected
if propagator:
    propagator.enable()
    # Set start date and step size
    startDate = 2440980.0
    stepSize = 60.0

    # Propagation loop
    for i in range(0,5):
            propagationTime = startDate + ((i*stepSize)/86400.0)
            propagator.propagate(population, propagationTime, stepSize)
            p = population.getPositionByIndex(0)
            print(p.x, p.y, p.z)

    # Disable propagator
    propagator.disable()
else:
    print("Propagator not found!")

# Clean up
del population
del propagator
