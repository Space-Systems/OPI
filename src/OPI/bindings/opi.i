%module OPI %{
#define OPI_CUDA_PREFIX
#include "opi_common.h"
#include "opi_types.h"
#include "opi_datatypes.h"
#include "opi_error.h"
#include "opi_host.h"
#include "opi_indexlist.h"
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
%include "opi_indexlist.h"
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
%array_functions(OPI::Epoch, epoch)

%extend OPI::JulianDay {
    OPI::JulianDay operator+(const long long& usec)
    {
        return OPI::operator+(OPI::JulianDay($self->day, $self->usec), usec);
    }
    OPI::JulianDay operator-(const long long& usec)
    {
        return OPI::operator-(OPI::JulianDay($self->day, $self->usec), usec);
    }
    OPI::JulianDay operator+(const OPI::JulianDay& b)
    {
        return OPI::operator+(OPI::JulianDay($self->day, $self->usec), b);
    }
    OPI::JulianDay operator-(const OPI::JulianDay& b)
    {
        return OPI::operator-(OPI::JulianDay($self->day, $self->usec), b);
    }
    bool operator>(const OPI::JulianDay b)
    {
        return OPI::operator>(*($self), b);
    }
    bool operator<(const OPI::JulianDay b)
    {
        return OPI::operator<(*($self), b);
    }
    bool operator>=(const OPI::JulianDay b)
    {
        return OPI::operator>=(*($self), b);
    }
    bool operator<=(const OPI::JulianDay b)
    {
        return OPI::operator<=(*($self), b);
    }
    bool operator==(const OPI::JulianDay b)
    {
        return OPI::operator==(*($self), b);
    }
}
// In python, the population getters are changed from C arrays to indexed setters and getters.
// So instead of getOrbit()[i] in C++, use getOrbitByIndex(i) in python.
// Alternatively, orbit_getitem(population.getOrbit(), i) will work.
// Instead of getOrbit()[i] = o, use setOrbit(i,o).

%extend OPI::Population {
        Orbit getOrbitByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getOrbit(device, no_sync)[index];
        }
        Vector3 getPositionByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getPosition(device, no_sync)[index];
        }
        Vector3 getVelocityByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getVelocity(device, no_sync)[index];
        }
        Vector3 getAccelerationByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getAcceleration(device, no_sync)[index];
        }
        Epoch getEpochByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getEpoch(device, no_sync)[index];
        }
        ObjectProperties getObjectPropertiesByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getObjectProperties(device, no_sync)[index];
        }
        Covariance getCovarianceByIndex(int index, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            return $self->getCovariance(device, no_sync)[index];
        }
        void setOrbit(int index, Orbit o, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getOrbit(device, no_sync)[index] = o;
        }
        void setPosition(int index, Vector3 pos, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getPosition(device, no_sync)[index] = pos;
        }
        void setVelocity(int index, Vector3 vel, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getVelocity(device, no_sync)[index] = vel;
        }
        void setAcceleration(int index, Vector3 acc, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getAcceleration(device, no_sync)[index] = acc;
        }
        void setEpoch(int index, Epoch ep, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getEpoch(device, no_sync)[index] = ep;
        }
        void setObjectProperties(int index, ObjectProperties props, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getObjectProperties(device, no_sync)[index] = props;
        }
        void setCovariance(int index, Covariance c, OPI::Device device=OPI::DEVICE_HOST, bool no_sync=false)
        {
            if (index < $self->getSize()) $self->getCovariance(device, no_sync)[index] = c;
        }
};
