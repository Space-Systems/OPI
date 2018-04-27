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
