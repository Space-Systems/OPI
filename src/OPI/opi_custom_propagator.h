#ifndef OPI_HOST_MODULED_PROPAGATOR_H
#define OPI_HOST_MODULED_PROPAGATOR_H

#include "opi_propagator.h"
#include <vector>
namespace OPI
{
	class PerturbationModule;
	class PropagatorIntegrator;

	struct CustomPropagatorImpl;

	//! \brief This class represents a propagator which can be composed from different perturbation modules and an integrator at runtime.
	//! \ingroup CPP_API_GROUP
	class CustomPropagator:
			public Propagator
	{
		public:
			//! Creates a new custom propagator with the specified name
            OPI_API_EXPORT CustomPropagator(const char* name);
			OPI_API_EXPORT ~CustomPropagator();
			/// Adds a module to this propagator
			OPI_API_EXPORT void addModule(PerturbationModule* module);
			/// Sets the integrator for this propagator
			OPI_API_EXPORT void setIntegrator(PropagatorIntegrator* integrator);

		protected:
			/// Override the propagation method
            virtual ErrorCode runPropagation(Population& population, double julian_day, double dt, PropagationMode mode = MODE_SINGLE_EPOCH, IndexList* indices = nullptr);
			virtual int requiresCUDA();

		private:
			Pimpl<CustomPropagatorImpl> impl;
	};
}

#endif
