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
#include "opi_module.h"

#include "opi_host.h"
#include <map>
#include <sstream>
namespace OPI
{
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	namespace
	{
		struct Property
		{
			char type;
			union
			{
				int* v_int;
				float* v_float;
				std::string* v_string;
			} ptr;
			Property():
				type(TYPE_UNKNOWN)
			{
				ptr.v_int = 0;
			}

			Property(int* value):
				type(TYPE_INTEGER)
			{
				ptr.v_int = value;
			}
			Property(float* value):
				type(TYPE_FLOAT)
			{
				ptr.v_float = value;
			}
			Property(std::string* value):
				type(TYPE_STRING)
			{
				ptr.v_string = value;
			}
		};
	}
	class ModuleImpl
	{
		public:
			Host* host;
			bool enabled;
			std::string name;
			std::string author;
			std::string description;
			std::map<std::string, Property> properties;
	};

	void Module::setHost(Host *newhost)
	{
		data->host = newhost;
	}
	//! \endcond

	Module::Module()
	{
		data = new ModuleImpl();
		data->host = 0;
		data->enabled = false;
	}

	Module::~Module()
	{
		delete data;
	}

	ErrorCode Module::enable()
	{
		// already enabled, do nothing
		ErrorCode status = NO_ERROR;
		if(!isEnabled())
			status = runEnable();
		if(status == NO_ERROR)
			data->enabled = true;
		data->host->sendError(status);
		return status;
	}

	ErrorCode Module::disable()
	{
		ErrorCode status = NO_ERROR;
		if(isEnabled())
		{
			status = runDisable();
			if(status == NO_ERROR)
				data->enabled = false;
		}
		data->host->sendError(status);
		return status;
	}

	ErrorCode Module::runEnable()
	{
		return NO_ERROR;
	}

	ErrorCode Module::runDisable()
	{
		return NO_ERROR;
	}

	bool Module::isEnabled() const
	{
		return data->enabled;
	}

	void Module::setName(const std::string &newname)
	{
		data->name = newname;
	}

	const std::string& Module::getName() const
	{
		return data->name;
	}

	void Module::setAuthor(const std::string& author)
	{
		data->author = author;
	}

	const std::string& Module::getAuthor() const
	{
		return data->author;
	}

	void Module::setDescription(const std::string& description)
	{
		data->description = description;
	}

	const std::string& Module::getDescription() const
	{
		return data->description;
	}

	Host* Module::getHost() const
	{
		return data->host;
	}

	void Module::registerProperty(const std::string &name, float *location)
	{
		data->properties.insert(std::make_pair(name, Property(location)));
	}

	void Module::registerProperty(const std::string &name, int *location)
	{
		data->properties.insert(std::make_pair(name, Property(location)));
	}

	void Module::registerProperty(const std::string &name, std::string *location)
	{
		data->properties.insert(std::make_pair(name, Property(location)));
	}

	void Module::setProperty(const std::string &name, int value)
	{
		// find property
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
					*(property.ptr.v_int) = value;
					break;
				case TYPE_FLOAT:
					*(property.ptr.v_float) = value;
					break;
				case TYPE_STRING:
				{
					std::stringstream stream;
					stream << value;
					*(property.ptr.v_string) = stream.str();
					break;
				}
			}
		}
	}
	void Module::setProperty(const std::string &name, float value)
	{
		// find property
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
					*(property.ptr.v_int) = value;
					break;
				case TYPE_FLOAT:
					*(property.ptr.v_float) = value;
					break;
				case TYPE_STRING:
				{
					std::stringstream stream;
					stream << value;
					*(property.ptr.v_string) = stream.str();
					break;
				}
			}
		}
	}
	void Module::setProperty(const std::string &name, const std::string& value)
	{
		// find property
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
					std::istringstream(value) >> *(property.ptr.v_int);
					break;
				case TYPE_FLOAT:
					std::istringstream(value) >> *(property.ptr.v_float);
					break;
				case TYPE_STRING:
				{
					*(property.ptr.v_string) = value;
					break;
				}
			}
		}
	}

	int Module::getPropertyInt(const std::string &name)
	{
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
					return *(property.ptr.v_int);
					break;
				case TYPE_FLOAT:
					return *(property.ptr.v_float);
					break;
				case TYPE_STRING:
				{
					int result;
					std::istringstream(*(property.ptr.v_string)) >> result;
					return result;
					break;
				}
			}
		}
		return 0;
	}

	float Module::getPropertyFloat(const std::string &name)
	{
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
					return *(property.ptr.v_int);
					break;
				case TYPE_FLOAT:
					return *(property.ptr.v_float);
					break;
				case TYPE_STRING:
				{
					float result;
					std::istringstream(*(property.ptr.v_string)) >> result;
					return result;
					break;
				}
			}
		}
		return 0.0f;
	}

	std::string Module::getPropertyString(const std::string &name)
	{
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_UNKNOWN:
					break;
				case TYPE_INTEGER:
				{
					std::stringstream stream;
					stream <<  *(property.ptr.v_int);
					return stream.str();
					break;
				}
				case TYPE_FLOAT:
				{
					std::stringstream stream;
					stream <<  *(property.ptr.v_float);
					return stream.str();
					break;
				}
				case TYPE_STRING:
				{
					return *(property.ptr.v_string);
					break;
				}
			}
		}
		return "";
	}

	int Module::getPropertyCount() const
	{
		return data->properties.size();
	}

	std::string Module::getPropertyName(int index) const
	{
		if((index < 0)||(index >= (int)(data->properties.size())))
			return "";
		std::map<std::string, Property>::iterator itr = data->properties.begin();
		for(int i=0; i < index; i++)
			itr++;
		return itr->first;
	}

	bool Module::hasProperty(const std::string &name) const
	{
		return data->properties.count(name) > 0;
	}

	PropertyType Module::getPropertyType(const std::string &name) const
	{
		if(data->properties.count(name) > 0)
			return data->properties[name].type;
		return TYPE_UNKNOWN;
	}
}
