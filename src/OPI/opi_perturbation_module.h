#ifndef OPI_PERTURBATION_MODULE_H
#define OPI_PERTURBATION_MODULE_H

#include "opi_common.h"
#include "opi_module.h"
#include "opi_perturbations.h"
#include "opi_error.h"
#include "opi_pimpl_helper.h"
namespace OPI
{
	class Population;
    class Perturbations;

	//! Contains the module implementation data
	class PerturbationModuleImpl;

	/*!
     * \brief This class represents a perturbation module which can be used by a Propagator
	 *
	 * \ingroup CPP_API_GROUP
	 * \see Module, Host
	 */
	class PerturbationModule: public Module
	{
		public:
			OPI_API_EXPORT PerturbationModule();
			OPI_API_EXPORT virtual ~PerturbationModule();
            //! Calculates the Perturbation for the passed dataset
			/**
             * The calculated perturbation forces will be added to the values present in delta
			 */
            OPI_API_EXPORT ErrorCode calculate(const Population& population, Perturbations& delta, JulianDay epoch, long long dt, PropagationMode mode = MODE_SINGLE_EPOCH, IndexList* indices = nullptr);

		protected:
            virtual ErrorCode runCalculation(const Population& population, Perturbations& delta, JulianDay epoch, long long dt, PropagationMode mode = MODE_SINGLE_EPOCH, IndexList* indices = nullptr);

		private:
			Pimpl<PerturbationModuleImpl> impl;
	};
}

#endif
