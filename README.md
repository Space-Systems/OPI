
OPI - Orbital Propagation Interface (2019 Version)
--------------------------------------------------

![OPI logo](https://raw.githubusercontent.com/ILR/OPI/master/logo/opi_logo_circular_small.png)

OPI is an interface with the goal to facilitate the implementation of orbital
propagators into different applications.

To calculate orbital motion, many different software programs exist emphasizing
on different aspects such as execution speed or accuracy. They often require
different input parameters and are written in different languages. This makes
comparing or exchanging them a challenging task. OPI aims at simplifying this
by providing a common way of handling propagation. Propagators using OPI are
designed as plugins/shared libraries that can be loaded by a host program via
the interface.

Features at a glance:
* Implement orbital propagators or force models as independent plugins
* Automatically find and load propagator plugins available on your platform
* Create, manage, copy and modify populations of orbital objects
* Multi-language support (C, C++, Fortran, Python, C#)
* Platform-independent (although hosts and propagators may not be)
* Extensible GPU computing support
* Automatic reading of configuration and resource files for propagators

OPI currently supports C, C++ and Fortran, as well as CUDA  and OpenCL for
propagators. The C API can also be used for integration into other languages
like Python or C#. Hosts and plugins don't have to be written in the same
language in order to collaborate. OPI itself is written in C++, with
auto-generated bindings for C and Fortran. For GPU support, it supplies a plugin
that scans for capable devices and helps to initialize CUDA or OpenCL-enabled
propagators.

Please note that this software is still under development and the interface
functions are subject to change. Your feedback is appreciated.


Usage (C++ Example)
-------------------

### Propagator

To implement a basic OPI propagator in C++, create a class that inherits from
OPI::Propagator. Apart from setting some basic constants you need to implement,
at the very least, its `runPropagation()` function:

```cpp
#include "OPI/opi_cpp.h"

// Set the name the plugin will appear as to the host
#define OPI_PLUGIN_NAME "MyPropagator"
// Set optional author and description
#define OPI_PLUGIN_AUTHOR "My Organization"
#define OPI_PLUGIN_DESC "Example Propagator"

// Set the version number for the plugin here.
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0

class MyPropagator: public OPI::Propagator
{
public:
  OPI::ErrorCode MyPropagator::runPropagation(OPI::Population& population, double julian_day, double dt,
    OPI::PropagationMode mode = OPI::MODE_SINGLE_EPOCH, OPI::IndexList* indices = nullptr);
  
  // Implement additional functions for returning information about the propagator,
  // loading proprietary file formats and defining init and de-init behaviour
};

// Include the macro that makes the propagator available as a plugin
#define OPI_IMPLEMENT_CPP_PROPAGATOR MyPropagator
#include "OPI/opi_implement_plugin.h"
```

The `runPropagation()` function is executed from the host application (via calls to `propagate()`), in a loop over all
propagation time steps. Its implementation inside the plugin manipulates the given population based on time information
as well as the state of the population itself. The following example also shows a way to handle index lists which can
optionally be provided by the host to specify which objects to consider:

```cpp
OPI::ErrorCode MyPropagator::runPropagation(OPI::Population& population, double julian_day, double dt,
    OPI::PropagationMode mode, OPI::IndexList* indices)
{
  // If an index list is given, loop over the size of the index list.
  // Otherwise, loop over the entire population.
  int loopSize = (indices ? indices->getSize() : population.getSize());

  for (int i=0; i<loopSize; i++)
  {
    // Get pointer to the Population's orbital data
    OPI::Orbit orbits = population.getOrbit();

    // Get object index from index list if given, or use loop counter otherwise
    int objectIndex = (indices ? indices->getData(OPI::DEVICE_HOST)[i] : i);

    // Calculate mean motion for the length of the given time step
    double meanMotion = sqrt(EARTH_GRAVITATIONAL_CONSTANT
      / pow(orbits[objectIndex].semi_major_axis, 3.0));

    // Add mean motion to the object's mean anomaly and normalize to radian range
    orbits[objectIndex].mean_anomaly =
      fmod(orbits[objectIndex].mean_anomaly + meanMotion * dt, 2*M_PI);
  }
  
  // Tell OPI that the population has been updated on the host device. This
  // ensures correct synchronization with CUDA and OpenCL devices.
  population.update(OPI::DATA_ORBIT, OPI::DEVICE_HOST);

  return OPI::SUCCESS;
}
```

If your propagator needs to read object data in a proprietary format, implement
the `loadPopulation()` function which is accessible from the host and allows you
to define the conversion to an OPI population. By default, it should read data in
OPI's own binary format.

```cpp
OPI::ErrorCode loadPopulation(OPI::Population& population, const char* filename)
{
  // Load given file or directory and fill population accordingly
  // The default is to simply load OPI's own binary format:
  population.read(filename);
  return OPI::SUCCESS;
}
```

### Host Application

To implement a basic host in C++, create a class that derives from OPI::Host, or
create an instance of it directly. Use it to load a plugin directory, select the
desired plugin and use it to propagate a population.

```cpp
int main(int argc, char* argv[])
{
  // Initialize host
  OPI::Host host;

  // Load plugin directory. Optionally, specify a GPU computing platform with
  // platform number and device number.
  host.loadPlugins("plugins",OPI::Host::PLATFORM_OPENCL, 0, 0);

  // Create a population with a single object on the given host
  OPI::Population population(host, 1);

  // Fill the population with orbit data
  population.getOrbit()[0] = { 6800.0, 0.0001, 23.5, 0.0, 0.0, 0.0 };

  // Get the desired propagator
  OPI::Propagator* myPropagator = host.getPropagator("MyPropagator");

  if (myPropagator)
  {
    // Initialize the propagator
    myPropagator->enable();

    // Set a start date and time step size for the propagation
    const double startDate = 2458201.5;
    const double stepSize = 60.0;

    // Propagate population for one day (1440 time steps at 60 seconds each)
    for (int i=0; i<1440; i++)
    {
      double currentTime = startDate + i * stepSize / 86400.0;
      // Propagate with the default settings (single epoch, no index list)
      OPI::ErrorCode status = myPropagator.propagate(populaion, currentTime, stepSize);
    }

    // Print the first object's mean anomaly
    std::cout << population.getOrbit()[0].mean_anomaly << endl;

    // Deinitialize the propagator
    myPropagator.disable();
  }

  return 0;
}
```

Changes From The 2015 Interface
-------------------------------

Over the last few years a few changes have been made to the interface that
have now been merged into the master branch. Those changes, dubbed the 2019
interface, deprecate the previous version and will require you to update
propagators and hosts. Updated documentation and examples will follow shortly, until
then the easiest way to get help at the moment is to contact me directly (mmoeckel
on GitHub). A quick overview of the most significant changes:

* Indexed propagation and multi-time propagation have moved from individual functions
to the main "propagate"/"calculate" functions. A mode setting has been introduced to switch
between single epoch (default) and individual epoch mode. If indexed propagation is not
required the IndexList pointer should be set to nullptr (default). Propagators that don't
support individual epoch mode or indexed propagation shall return NOT_IMPLEMENTED when
IndexList is not null or individual epoch mode is selected, respectively. To update from
the 2015 interface, simply implement your runPropagation function like this:
```cpp
OPI::ErrorCode MyPropagator::runPropagation(OPI::Population& population, double julian_day, double dt, OPI::PropagationMode mode, OPI::IndexList* indices)
{
  if (mode == OPI::MODE_INDIVIDUAL_EPOCHS)
  {
    // move code from runMultiTimePropagation() here, or:
    return OPI::NOT_IMPLEMENTED;
  }

  if (indices != nullptr)
  {
    // move code from runIndexedPropagation() here, or:
    return OPI::NOT_IMPLEMENTED;
  }

  // implement propagation function
  return OPI::SUCCESS;
}
```
* Beginning of life and end of life fields have moved to an "Epoch" struct inside
the population. The Epoch also has a "current_epoch" field that is used for individual
epoch propagation.
* Perturbation modules now return a Perturbation instance containing delta values instead
of a modified population:
```cpp
OPI::ErrorCode MyForceModel::runCalculation(OPI::Population& data, OPI::Perturbations& delta, double julian_day, double dt, OPI::PropagationMode mode, OPI::IndexList* indices)
{
  // Population is read-only
  // Calculate changes and store them in the appropriate field in "delta"
}
```
* Covariance matrix fields have been introduced.
* Propagators can implement a "loadPopulation" function that loads propagator-specific
files and creates OPI populations from it. The default behaviour of this function is to
load a population in the binary .opi format.
* Propagators now have an "align" function that brings all objects to the same epoch
(that of its "latest" object). It should automatically work with all propagators that
support both individual epoch mode and indexed propagation.
* New functions to copy and append populations.
* Saved populations (.opi files) are now stored gzipped.
* All modules can now read config files individually.

Issues:
* Example code is outdated
* Fortran interface is untested




FAQ
---

Q: Does OPI's open source license affect my host and propagator applications?

A: No. OPI is explicitly designed to be independent from hosts and propagators
so using OPI does not require you to adopt open source licenses for your
applications or plugins. While we always welcome all kinds of source code
contributions we only ask you to submit fixes and improvements that you
make to the OPI library itself. One major goal of OPI is to be able to exchange,
compare and collaborate on orbital propagators with other researchers. If you are
unable to share your propagator or force model implementation in source code form
then OPI may provide an easy way for you to distribute it as a pre-compiled plugin
instead.



Building Instructions
---------------------

Building OPI has been tested on Linux (recent versions of Debian, Ubuntu and
OpenSuSE), Windows (Visual Studio 2017/2019) and OSX (deprecated). OPI uses
CMake as a build system, so simply follow the usual instructions below or use
the GUI tool (in-place builds are not allowed):

```
mkdir build
cd build
cmake .. #or 'cmake-gui ..' - whichever you prefer
make
make install
make doc #optional, to build the API documentation - requires Doxygen
```

You can set the CMAKE_INSTALL_PREFIX variable to a custom directory of you
don't want a system-wide install. In that case, you must make sure that the
lib directory is in your library path at runtime by setting the LD_LIBRARY_PATH
variable accordingly. If you require support for CUDA propagators, make sure
the CUDA SDK is installed and can be found by CMake.

To start using OPI, take a look at the documentation provided with the library.
If you have any questions, please contact me (mmoeckel on GitHub).
