/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
#ifndef OPI_MODULE_H
#define OPI_MODULE_H
#include "opi_common.h"
#include "opi_data.h"
#include "opi_error.h"
#include <string>
namespace OPI
{
	class ObjectData;
	class IndexList;

	//! Contains the module implementation data
	class ModuleImpl;

	//! \brief This interface class defines the common module shared functions
	//! \ingroup CPP_API_GROUP
	class OPI_API_EXPORT Module
	{
		public:
			Module();
			virtual ~Module();

			//! Checks if this propagator is enabled
			bool isEnabled() const;
			//! Enable this propagator (reserves internal memory)
			ErrorCode enable();
			//! Disable this propagator (frees up internal memory)
			ErrorCode disable();
			//! Sets the name of this module
			void setName(const std::string& name);
			//! Returns the name of this module
			const std::string& getName() const;
			//! Sets the author of this module
			void setAuthor(const std::string& name);
			//! Returns the author of this module
			const std::string& getAuthor() const;
			//! Sets the description of this module
			void setDescription(const std::string& name);
			//! Returns the description of this module
			const std::string& getDescription() const;

			//! Sets the version number of this module
			void setVersion(int major, int minor, int patch);
			//! Gets the major version of this module
			int getVersionMajor() const;
			//! Gets the minor version of this module
			int getVersionMinor() const;
			//! Gets the patch version of this module
			int getVersionPatch() const;

			//! registers a property
			void registerProperty(const std::string& name, int* location);
			//! registers a property
			void registerProperty(const std::string& name, float* location);
			//! registers a property
			void registerProperty(const std::string& name, std::string* location);
			//! Sets a property
			void setProperty(const std::string& name, int value);
			//! Sets a property
			void setProperty(const std::string& name, float value);
			//! Sets a property
			void setProperty(const std::string& name, const std::string& value);
			//! Gets the value of a given property
			int getPropertyInt(const std::string& name);
			//! Gets the value of a given property
			float getPropertyFloat(const std::string& name);
			//! Gets the value of a given property
			std::string getPropertyString(const std::string& name);

			//! Returns the amount of registered properties
			int getPropertyCount() const;
			//! Returns the name of the property identified by the given index
			std::string getPropertyName(int index) const;

			//! Checks if a property is registered
			bool hasProperty(const std::string& name) const;
			//! Returns the type of a property
			PropertyType getPropertyType(const std::string& name) const;
		protected:
			//! Returns the Host of this propagator
			Host* getHost() const;
			//! Override this if you want to change the enable behaviour
			virtual ErrorCode runEnable();
			//! Override this if you want to change the disable behaviour
			virtual ErrorCode runDisable();

		private:
			//! \cond INTERNAL_DOCUMENTATION
			void setHost(OPI::Host* newhost);
			friend class Host;
			//! \endcond
			ModuleImpl* data;
	};
}

#endif
