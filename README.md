
OPI - Orbital Propagation Interface (2019 Version)
--------------------------------------------------

OPI is an interface with the goal to facilitate the implementation of orbital
propagators into different applications.

To calculate orbital motion, many different software programs exist emphasizing
on different aspects such as execution speed or accuracy. They often require
different input parameters and are written in different languages. This makes
comparing or exchanging them a challenging task. OPI aims at simplifying this
by providing a common way of handling propagation. Propagators using OPI are
designed as plugins/shared libraries that can be loaded by a host program via
the interface.

OPI currently supports C, C++ and Fortran, as well as CUDA  and OpenCL for
propagators. The C API can also be used for integration into other languages
like Python or C#. Hosts and plugins don't have to be written in the same
language in order to collaborate. OPI itself is written in C++, with
auto-generated bindings for C and Fortran. For GPU support, it supplies a plugin
that scans for capable devices and helps to initialize CUDA or OpenCL-enabled
propagators.

Please note that this software is still under development and the interface
functions are subject to change. Your feedback is appreciated.


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



BUILD INSTRUCTIONS
------------------

Building OPI has been tested on Linux (recent versions of Debian, Ubuntu and
OpenSuSE), Windows (Visual Studio 2017/2019) and OSX (deprecated). OPI uses
CMake as a build system, so simply follow the usual instructions below or use
the GUI tool (in-place builds are not allowed):

mkdir build
cd build
cmake .. #or 'cmake-gui ..' - whichever you prefer
make
make install
make doc #optional, to build the API documentation - requires Doxygen

You can set the CMAKE_INSTALL_PREFIX variable to a custom directory of you
don't want a system-wide install. In that case, you must make sure that the
lib directory is in your library path at runtime by setting the LD_LIBRARY_PATH
variable accordingly. If you require support for CUDA propagators, make sure
the CUDA SDK is installed and can be found by CMake.

To start using OPI, take a look at the documentation provided with the library.
The example code is quite outdated and will be updated shortly.
If you have any questions, please contact me (mmoeckel on GitHub).
