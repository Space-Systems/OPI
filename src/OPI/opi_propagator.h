#ifndef OPI_PROPAGATOR_H
#define OPI_PROPAGATOR_H
#include "opi_common.h"
#include "opi_data.h"
#include "opi_error.h"
#include "opi_module.h"
#include <string>
namespace OPI
{
	class ObjectData;
	class IndexList;

	//! Contains the propagation implementation data
	class PropagatorImpl;

	//! \brief This interface class defines the way to operate on the
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT Propagator:
			public Module
	{
		public:
			Propagator();
			virtual ~Propagator();

			//! Starts the propagation for the given time frame
			ErrorCode propagate(ObjectData& data, float years, float seconds, float dt );
			//! Starts the index-based propagation for the given time frame
			ErrorCode propagate(ObjectData& data, IndexList& indices, float years, float seconds, float dt);

			//! Check if this propagator is able to propagate backwards
			virtual bool backwardPropagation();

		protected:
			//! The actual propagation implementation
			virtual ErrorCode runPropagation(ObjectData& data, float years, float seconds, float dt ) = 0;
			//! Override this to implement an index-based propagation
			virtual ErrorCode runIndexedPropagation(ObjectData& data, IndexList& indices, float years, float seconds, float dt );

		private:
			PropagatorImpl* data;
	};
}

#endif
