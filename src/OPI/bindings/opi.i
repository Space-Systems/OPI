%module OPI %{
#define OPI_CUDA_PREFIX
#include "opi_common.h"
#include "opi_types.h"
#include "opi_datatypes.h"
#include "opi_error.h"
#include "opi_host.h"
#include "opi_population.h"
#include "opi_module.h"
#include "opi_propagator.h"
#include "opi_custom_propagator.h"
#include "opi_perturbation_module.h"
%}

%include <std_string.i>

#define OPI_CUDA_PREFIX
%include "opi_common.h"
%include "opi_types.h"
%include "opi_datatypes.h"
%include "opi_error.h"
%include "opi_host.h"
%include "opi_population.h"
%include "opi_module.h"
%include "opi_propagator.h"
%include "opi_custom_propagator.h"
%include "opi_perturbation_module.h"

%include <carrays.i>
%array_functions(OPI::Orbit, orbit)
%array_functions(OPI::Vector3, vector3)
%array_functions(OPI::ObjectProperties, props)
%array_functions(OPI::Covariance, covariance)

// In python, the population getters are changed from C arrays to indexed setters and getters.
// So instead of getOrbit()[i] in C++, use getOrbit(i) in python.
// Instead of getOrbit()[i] = o, use setOrbit(o,i).

%feature("shadow") Population::getOrbit() %{
def getOrbit(index):
  return orbit_getitem(getOrbit(),index)
%}

%feature("shadow") Population::getPosition() %{
def getPosition(index):
  return vector3_getitem(getPosition(),index)
%}

%feature("shadow") Population::getVelocity() %{
def getVelocity(index):
  return vector3_getitem(getVelocity(),index)
%}

%feature("shadow") Population::getAcceleration() %{
def getAcceleration(index):
  return vector3_getitem(getAcceleration(),index)
%}

%feature("shadow") Population::getObjectProperties() %{
def getObjectProperties(index):
  return props_getitem(getObjectProperties(),index)
%}

%feature("shadow") Population::getCovariance() %{
def getCovariance(index):
  return covariance_getitem(getCovariance(),index)
%}

%extend OPI::Population {
        void setOrbit(Orbit o, int index) { if (index < $self->getSize()) $self->getOrbit()[index] = o; }
        void setPosition(Vector3 pos, int index) { if (index < $self->getSize()) $self->getPosition()[index] = pos; }
        void setVelocity(Vector3 vel, int index) { if (index < $self->getSize()) $self->getVelocity()[index] = vel; }
        void setAcceleration(Vector3 acc, int index) { if (index < $self->getSize()) $self->getAcceleration()[index] = acc; }
        void setObjectProperties(ObjectProperties props, int index) { if (index < $self->getSize()) $self->getObjectProperties()[index] = props; }
        void setCovariance(Covariance c, int index) { if (index < $self->getSize()) $self->getCovariance()[index] = c; }
};
