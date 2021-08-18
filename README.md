
OPI - Orbital Propagation Interface (2021 Version)
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

OPI currently supports C, C++, Fortran, CUDA and OpenCL for propagators, and
C, C++, Fortran and Python for hosts. The C API can also be used for integration
into other languages like C#. Hosts and plugins don't have to be written in the
same language in order to collaborate. OPI itself is written in C++, with
auto-generated bindings for C, Fortran and Python. For GPU support, it supplies
a plugin that scans for capable devices and helps to initialize CUDA or
OpenCL-enabled propagators.

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

### PropagatorProperties and Resources

Propagators can define configuration variables called `PropagatorProperties` that
can be set by the host program at runtime, or via a config file. Supported variable
types are floats/doubles, integers and strings. To declare a PropagatorProperty inside the
Propagator, use one of the functions `registerProperty()` or `createProperty()`. The
former will let you bind an existing variable to a Property while the latter will
let OPI manage the variable internally:

```cpp
class PropertiesCPP: public OPI::Propagator
{
private:
    int testproperty_int;
    float testproperty_float;
    std::string testproperty_string;

public:
    PropertiesCPP(OPI::Host& host)
    {
        testproperty_int = 0;
        testproperty_float = 0.0f;
        testproperty_string = "test";

        // Expose the above member variables as properties
        registerProperty("ThisIsAnInteger", &testproperty_int);
        registerProperty("ThisIsAFloat", &testproperty_float);
        registerProperty("ThisIsAString", &testproperty_string);

        // This creates a string property that's managed by OPI.
        createProperty("ThisIsAnotherString", "defaultString");
    }
```

In order to access Properties from either the host or the propagator, use the appropriate
setter and getter functions:

```cpp
setProperty("ThisIsAnInteger", 23);
int value = getPropertyInt("ThisIsAnInteger");
```

Another way to set properties and assign default values is via a config file. Simply
place a file with the same base name and the `.cfg` suffix into the plugin folder and OPI
will read the properties from there (properties not defined by the propagator will be
automatically added using the `createProperty()` function). For example, if your plugin is
`plugins/sgp4.dll`, create a file named `plugins/sgp4.cfg` with the following contents:

```
# "Operations Mode" - "a" for advanced (default), "i" for improved.
opsmode = "a"

# Gravity constants - "wgs72" (default), "wgs72old" or "wgs84".
whichconst = "wgs72"
```

This will create the string properties `opsmode` and `whichconst` and set them to the
given default values. Data types in config files are denoted by quotation marks and decimal
dots (i.e. 1.0 would be parsed as a float, 1 as an integer and "1.0" as a string).

Most propagators rely on additional data in order to function. Similar to config files, OPI
Propagators can read data from resource archives. If you place a zip archive containing your
resource files into the plugin folder with the plugin's base name and the suffix `.dat`, you
can access the files inside that archive from the propagator. For example, if your plugin
is `plugins/sgp4.dll`, create a zip archive containing the file "test.txt" and place it as
`plugins/sgp4.dat`. Then, inside the propagator, simply access its contents as follows:

```cpp
char* buffer;
size_t size = loadResource("test.txt", &buffer);
std::cout << buffer << std::endl;
delete buffer; // Don't forget to free the buffer after you're done with it.
```

### Host Application

To implement a basic host in C++, create a class that derives from OPI::Host, or
create an instance of it directly. Use it to load a plugin directory, select the
desired plugin and use it to propagate a population.

```cpp
#define OPI_DISABLE_OPENCL //OpenCL not required for host
#include "OPI/opi_cpp.h"

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

    // Set a start date for the propagation.
    const OPI::JulianDay startDate = OPI::fromDouble(2458201.5);
    // Set the propagation time step in microseconds.
    const long stepSize = 60000000l;

    // Propagate population for one day (1440 time steps at 60 seconds each)
    for (int i=0; i<1440; i++)
    {
      // The + operator will add microseconds to a JulianDay.
      OPI::JulianDay currentTime = startDate + i * stepSize;
      // Propagate with the default settings (single epoch, no index list)
      OPI::ErrorCode status = myPropagator.propagate(population, currentTime, stepSize);
    }

    // Print the first object's mean anomaly
    std::cout << population.getOrbit()[0].mean_anomaly << endl;

    // Deinitialize the propagator
    myPropagator.disable();
  }

  return 0;
}
```

Changes From The 2019 Interface
-------------------------------

It was found that expressing Julian dates as double precision floats can introduce
inaccuracies in some cases, especially when propagating with very small time steps
(e.g. milliseconds). In the above example, the OPI 2019 host would have incremented
the propagation time step like this:

```cpp
double startDate = 2458201.5;
double stepSize = 60.0; //seconds
double propagationTime = startDate + (i*stepSize)/86400.0;
OPI::ErrorCode status = myPropagator.propagate(population, currentTime, stepSize);
```

In multi-epoch mode, currentTime would be ignored and the propagator would take care
of time-keeping, usually like this:

```cpp
population.getEpoch()[i].current_epoch += (dt/86400.0);
```

The floating point inaccuracies introduced by these two different methods of incrementing
the time can lead to different results depending on which mode is used. In order to enable
OPI to provide guaranteed microsecond accuracy, the OPI::JulianDay format was introduced. It
breaks up the Julian date into two components:

```cpp
struct JulianDay {
  int day; // Full days (integer component of the Julian day)
  long usec; // Fraction of the day in microseconds
};
```

All Epoch fields of OPI::Population have been updated to use this format. The propagate()
function will now accept a JulianDay for the propagation epoch, and a long for the time
step; time step units have changed from seconds to microseconds. Functions have been added
to convert between JulianDay and double, however, since these may introduce the type of
inaccuracies described above they should be used only when absolutely necessary:

```cpp
double inputDate = 2458201.5;
OPI::JulianDay propagationDate = OPI::fromDouble(inputDate);
double outputDate = OPI::toDouble(propagationDate);
```

Other changes:
* Make Population const in PerturbationModule
* C interface function renamed from `propagateAll` to `propagate`


Changes From The 2015 Interface
-------------------------------

Over the last few years a few changes have been made to the interface that
have now been merged into the master branch. Those changes, dubbed the 2019
interface, deprecate the previous version and will require you to update
propagators and hosts. Updated documentation and examples will follow shortly, until
then the easiest way to get help at the moment is to contact me directly (mmoeckel
on GitHub). A quick overview of the most significant changes:

* New license: Moved from LGPL to the simpler and more permissive MIT license.
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
* Support for reading files from resource archives.

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


OPI Population Data Format (OPI-2019 and above)
-----------------------------------------------

OPI Populations can be read from and written to files using the functions
Population::read() and Population::write(), respectively. The resulting files,
commonly given the suffix ".opi", contain zlib-compressed binary data. When
uncompressed, the data is divided into three sections: A header, a population
info block, and an object info block. (Note that all integers are signed unless
otherwise specified.)

The header contains three values:
* A 32-bit integer containing the magic number 47627. This is used for identification.
* A 32-bit integer containing the file format revision number. This number is incremented
 with every OPI release that makes changes to the file format and is used to provide
 backwards compatibility with files that have been written using older versions of OPI.
* A 32-bit integer containing the population size (i.e. number of objects in the population).
 This is used for correct interpretation of data in the object info block.

The population info block contains three value pairs, each consisting of a 32-bit integer
containing the length of the following character string, and the character string itself,
holding the actual information. The values are:
* Name of the propagator that this population was last propagated with, or "None".
* A user-defined population description (format revision number 2 or higher). The description
 can be empty, denoted by a length of zero bytes.
* A string representation of the reference frame the population is in (format revision
 number 3 or higher).

The object info block starts with object names. These consist of a number of 32-bit
integer/character string pairs corresponding to the population size:
* A 32-bit integer containing the length of the string for the first object name, or
 zero if no object name is given,
* if the length is larger than zero, a corresponding number of characters containing the
 name of the first object.
* Repeat for population size.

Next follow blocks of object data (orbits, properties, state vectors, and so on),
each with the following format:
* A 32-bit integer containing the block identification number,
* A 32-bit integer containing the size of the corresponding data type,
* The vector containing the actual object data.

The block identification number is necessary because not
every population uses all data types (e.g. some may use orbits and no state vectors while
others use state vectors exclusively, and some may have both types set), so the data blocks
do not have a fixed order. Instead, the block identification number states which type of data
is next in the file. It corresponds to the DataType enum:
```
DATA_ORBIT = 0,
DATA_PROPERTIES = 1,
DATA_POSITION = 2,
DATA_VELOCITY = 3,
DATA_ACCELERATION = 4,
DATA_EPOCH = 5,
DATA_COVARIANCE = 6,
DATA_BYTES = 7
```
Following the block identification number is another 32-bit integer containing the size
of the corresponding data type (OPI::Orbit, OPI::Vector3, and so on). If the block ID is
7 (DATA_BYTES) then this integer contains the number of bytes allocated for each object.
With this information, the data vector can be interpreted. Its size is given by the population
size from the header multiplied by the data type size. For example, if the block ID is 2, the
block contains position data. Since this is of type OPI::Vector3, the next integer will be 24
(the size of three 64-bit doubles). If the population size given in the header is 10 objects then
the next 10 x 24 = 240 bytes need to be read into the population's position vector.

Data type sizes can change between file format revisions. For example, revisions 4
and 5 each add one field to the Epoch type while revision 6 changes the format in which
Epoch fields are stored from double to the JulianDay type. In other words, the size of
the Epoch data type is:
* three doubles in revision 3 and lower,
* four doubles in revision 4,
* five doubles in revision 5,
* five pairs of two 64-bit long integers in revision 6 and up.


Graphical User Interface
------------------------

For a graphical OPI host and population editor, check out [OPI Explorer](https://github.com/Space-Systems/OPI-Explorer).
