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
			// enum describing the property base type
			char type;
			union
			{
				int* v_int;
				float* v_float;
				std::string* v_string;
			} ptr;
			// true if the object is allocated by OPI
			bool allocated;
			Property():
				type(TYPE_UNKNOWN)
			{
				ptr.v_int = 0;
				allocated = false;
			}

			Property(int* value):
				type(TYPE_INTEGER)
			{
				ptr.v_int = value;
				allocated = false;
			}
			Property(int value):
				type(TYPE_INTEGER)
			{
				ptr.v_int = new int;
				*ptr.v_int = value;
				allocated = true;
			}

			Property(float* value):
				type(TYPE_FLOAT)
			{
				ptr.v_float = value;
				allocated = false;
			}

			Property(float value):
				type(TYPE_FLOAT)
			{
				ptr.v_float = new float;
				*ptr.v_float = value;
				allocated = true;
			}
			Property(std::string* value):
				type(TYPE_STRING)
			{
				ptr.v_string = value;
				allocated = false;
			}
			Property(const std::string& value):
				type(TYPE_STRING)
			{
				ptr.v_string = new std::string;
				*ptr.v_string = value;
				allocated = true;
			}
			~Property()
			{
				if(allocated)
				{
					if(type == TYPE_INTEGER)
						delete ptr.v_int;
					else if(type == TYPE_FLOAT)
						delete ptr.v_float;
					else if(type == TYPE_STRING)
						delete ptr.v_string;
				}
			}
			Property(const Property& other)
			{
				if(other.allocated)
				{
					if(other.type == TYPE_INTEGER)
						ptr.v_int = new int(*(other.ptr.v_int));
					else if(other.type == TYPE_FLOAT)
						ptr.v_float = new float(*(other.ptr.v_float));
					else if(other.type == TYPE_STRING)
						ptr.v_string = new std::string(*(other.ptr.v_string));
				}
				else
					ptr = other.ptr;
				allocated = other.allocated;
				type = other.type;
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
			void* privateData;
	};

	void Module::setHost(Host *newhost)
	{
		data->host = newhost;
	}
	//! \endcond

	Module::Module()
	{
		data->host = 0;
		data->enabled = false;
	}

	Module::~Module()
	{
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

	void Module::createProperty(const std::string &name, int value)
	{
		data->properties.insert(std::make_pair(name, Property(value)));
	}

	void Module::createProperty(const std::string &name, float value)
	{
		data->properties.insert(std::make_pair(name, Property(value)));
	}

	void Module::createProperty(const std::string &name, const std::string& value)
	{
		data->properties.insert(std::make_pair(name, Property(value)));
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

	const std::string& Module::getPropertyString(const std::string &name)
	{
		static std::string internal_bufferstring;
		internal_bufferstring = "";
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
					internal_bufferstring = stream.str();
					break;
				}
				case TYPE_FLOAT:
				{
					std::stringstream stream;
					stream <<  *(property.ptr.v_float);
					internal_bufferstring = stream.str();
					break;
				}
				case TYPE_STRING:
				{
					internal_bufferstring =  *(property.ptr.v_string);
					break;
				}
			}
		}		
		return internal_bufferstring;
	}

	void Module::setPrivateData(void* private_data)
	{
		data->privateData = private_data;
	}

	void* Module::getPrivateData()
	{
		return data->privateData;
	}

	int Module::getPropertyCount() const
	{
		return data->properties.size();
	}

	const std::string& Module::getPropertyName(int index) const
	{
		static std::string dummystring = "";
		if((index < 0)||(index >= (int)(data->properties.size())))
			return dummystring;
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
