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
			PropertyType type;
			int size;
			union
			{
				int* v_int;
				float* v_float;
				double* v_double;
				std::string* v_string;
			} ptr;
			// true if the object is allocated by OPI
			bool allocated;
			Property(): type(TYPE_UNKNOWN)
			{
				ptr.v_int = 0;
				size = 0;
				allocated = false;
			}

			Property(int* value, bool allocate, int array_size = 1):
				type((array_size == 1) ? TYPE_INTEGER : TYPE_INTEGER_ARRAY)
			{
				size = array_size;
				allocated = allocate;
				if(allocate)
				{
					if(type == TYPE_INTEGER)
						ptr.v_int = new int(*value);
					else
					{
						ptr.v_int = new int[size];
						for(int i = 0; i < size; ++i)
							ptr.v_int[i] = value[i];
					}
				}
				else
					ptr.v_int = value;
			}

			Property(float* value, bool allocate, int array_size = 1):
				type((array_size == 1) ? TYPE_FLOAT : TYPE_FLOAT_ARRAY)
			{
				size = array_size;
				allocated = allocate;
				if(allocate)
				{
					if(type == TYPE_FLOAT)
						ptr.v_float = new float(*value);
					else
					{
						ptr.v_float = new float[size];
						for(int i = 0; i < size; ++i)
							ptr.v_float[i] = value[i];
					}
				}
				else
					ptr.v_float = value;
			}

			Property(double* value, bool allocate, int array_size = 1):
				type((array_size == 1) ? TYPE_DOUBLE : TYPE_DOUBLE_ARRAY)
			{
				size = array_size;
				allocated = allocate;
				if(allocate)
				{
					if(type == TYPE_DOUBLE)
						ptr.v_double = new double(*value);
					else
					{
						ptr.v_double = new double[size];
						for(int i = 0; i < size; ++i)
							ptr.v_double[i] = value[i];
					}
				}
				else
					ptr.v_double = value;
			}

			Property(std::string* value):
				type(TYPE_STRING)
			{
				size = 1;
				allocated = false;
				ptr.v_string = value;
			}

			Property(const std::string& value):
				type(TYPE_STRING)
			{
				size = 1;
				allocated = true;
				ptr.v_string = new std::string(value);
			}

			~Property()
			{
				if(allocated)
				{
					switch(type)
					{
						case TYPE_INTEGER:
							delete ptr.v_int;
							break;
						case TYPE_FLOAT:
							delete ptr.v_float;
							break;
						case TYPE_DOUBLE:
							delete ptr.v_double;
							break;
						case TYPE_STRING:
							delete ptr.v_string;
							break;
						case TYPE_INTEGER_ARRAY:
							delete[] ptr.v_int;
							break;
						case TYPE_FLOAT_ARRAY:
							delete[] ptr.v_float;
							break;
						case TYPE_DOUBLE_ARRAY:
							delete[] ptr.v_double;
							break;
						case TYPE_UNKNOWN:
							break;
					}
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
					else if(other.type == TYPE_DOUBLE)
						ptr.v_double = new double(*(other.ptr.v_double));
					else if(other.type == TYPE_STRING)
						ptr.v_string = new std::string(*(other.ptr.v_string));
					else if(other.type == TYPE_INTEGER_ARRAY)
					{
						size = other.size;
						ptr.v_int = new int[size];
						for(int i = 0; i < size; ++i)
							ptr.v_int[i] = other.ptr.v_int[i];
					}
					else if(other.type == TYPE_FLOAT_ARRAY)
					{
						size = other.size;
						ptr.v_float = new float[size];
						for(int i = 0; i < size; ++i)
							ptr.v_float[i] = other.ptr.v_float[i];
					}
					else if(other.type == TYPE_DOUBLE_ARRAY)
					{
						size = other.size;
						ptr.v_double = new double[size];
						for(int i = 0; i < size; ++i)
							ptr.v_double[i] = other.ptr.v_double[i];
					}
				}
				else
					ptr = other.ptr;
				allocated = other.allocated;
				type = other.type;
				size = other.size;
			}

			template<class T>
			void assignString(const T& value)
			{
				std::stringstream stream;
				stream << value;
				*(ptr.v_string) = stream.str();
			}


			template<class T>
			ErrorCode setValue(const T& value)
			{
				ErrorCode status = NO_ERROR;
				switch(type)
				{
					case TYPE_INTEGER:
						*(ptr.v_int) = value;
						break;
					case TYPE_FLOAT:
						*(ptr.v_float) = value;
						break;
					case TYPE_DOUBLE:
						*(ptr.v_double) = value;
						break;
					case TYPE_STRING:
						assignString(value);
						break;
					case TYPE_INTEGER_ARRAY:
					case TYPE_FLOAT_ARRAY:
					case TYPE_DOUBLE_ARRAY:
					case TYPE_UNKNOWN:
						status = INCOMPATIBLE_TYPES;
						break;
				}
				return status;
			}
			template<class T>
			ErrorCode setValue(T* values, int n)
			{
				ErrorCode status = NO_ERROR;
				if( n <= 0 )
					status = INDEX_RANGE;
				else if( n == 1)
					status = setValue(values[0]);
				else if( size != n)
					status = INCOMPATIBLE_TYPES;
				else
				{
					switch(type)
					{
						case TYPE_INTEGER_ARRAY:
							for(int i = 0; i < size; ++i)
								ptr.v_int[i] = values[i];
							break;
						case TYPE_FLOAT_ARRAY:
							for(int i = 0; i < size; ++i)
								ptr.v_float[i] = values[i];
							break;
						case TYPE_DOUBLE_ARRAY:
							for(int i = 0; i < size; ++i)
								ptr.v_double[i] = values[i];
							break;
						case TYPE_INTEGER:
						case TYPE_FLOAT:
						case TYPE_DOUBLE:
						case TYPE_STRING:
						case TYPE_UNKNOWN:
							status = INCOMPATIBLE_TYPES;
							break;
					}
				}
				return status;
			}

			template<class T>
			ErrorCode getValue(T& value, int element)
			{
				switch(type)
				{
					case TYPE_INTEGER:
						if(element != 0) return INDEX_RANGE;
						value = *(ptr.v_int);
						break;
					case TYPE_FLOAT:
						if(element != 0) return INDEX_RANGE;
						value = *(ptr.v_float);
						break;
					case TYPE_DOUBLE:
						if(element != 0) return INDEX_RANGE;
						value = *(ptr.v_double);
						break;
					case TYPE_STRING:
						if(element != 0) return INDEX_RANGE;
						std::istringstream(*(ptr.v_string)) >> value;
						break;
					case TYPE_INTEGER_ARRAY:
						if((element < 0 || element > size)) return INDEX_RANGE;
						value = ptr.v_int[element];
						break;
					case TYPE_FLOAT_ARRAY:
						if((element < 0 || element > size)) return INDEX_RANGE;
						value = ptr.v_float[element];
						break;
					case TYPE_DOUBLE_ARRAY:
						if((element < 0 || element > size)) return INDEX_RANGE;
						value = ptr.v_double[element];
						break;
					case TYPE_UNKNOWN:
						return INCOMPATIBLE_TYPES;
				}
				return NO_ERROR;
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
			template<class T> ErrorCode setValue(const std::string& name, const T& value);
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

	void Module::registerProperty(const std::string &name, int *location)
	{
		data->properties.insert(std::make_pair(name, Property(location, false)));
	}

	void Module::registerProperty(const std::string &name, float *location)
	{
		data->properties.insert(std::make_pair(name, Property(location, false)));
	}

	void Module::registerProperty(const std::string &name, double *location)
	{
		data->properties.insert(std::make_pair(name, Property(location, false)));
	}

	void Module::registerProperty(const std::string &name, std::string *location)
	{
		data->properties.insert(std::make_pair(name, Property(location)));
	}

	void Module::registerProperty(const std::string &name, int *location, int size)
	{
		data->properties.insert(std::make_pair(name, Property(location, false, size)));
	}

	void Module::registerProperty(const std::string &name, float *location, int size)
	{
		data->properties.insert(std::make_pair(name, Property(location, false, size)));
	}

	void Module::registerProperty(const std::string &name, double *location, int size)
	{
		data->properties.insert(std::make_pair(name, Property(location, false, size)));
	}


	void Module::createProperty(const std::string &name, int value)
	{
		data->properties.insert(std::make_pair(name, Property(&value, true)));
	}

	void Module::createProperty(const std::string &name, float value)
	{
		data->properties.insert(std::make_pair(name, Property(&value, true)));
	}

	void Module::createProperty(const std::string &name, double value)
	{
		data->properties.insert(std::make_pair(name, Property(&value, true)));
	}

	void Module::createProperty(const std::string &name, const std::string& value)
	{
		data->properties.insert(std::make_pair(name, Property(value)));
	}

	template<class T>
	ErrorCode ModuleImpl::setValue(const std::string& name, const T& value)
	{
		ErrorCode result = INVALID_PROPERTY;
		// find property
		if(properties.count(name) > 0)
		{
			result = properties[name].setValue(value);
		}
		host->sendError(result);
		return NO_ERROR;
	}

	ErrorCode Module::setProperty(const std::string &name, int value)
	{
		return data->setValue(name, value);
	}
	ErrorCode Module::setProperty(const std::string &name, float value)
	{
		return data->setValue(name, value);
	}
	ErrorCode Module::setProperty(const std::string &name, double value)
	{
		return data->setValue(name, value);
	}

	ErrorCode Module::setProperty(const std::string &name, const std::string& value)
	{
		ErrorCode result = INVALID_PROPERTY;
		// find property
		if(data->properties.count(name) > 0)
		{
			result = NO_ERROR;
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
				case TYPE_INTEGER:
					std::istringstream(value) >> *(property.ptr.v_int);
					break;
				case TYPE_FLOAT:
					std::istringstream(value) >> *(property.ptr.v_float);
					break;
				case TYPE_DOUBLE:
					std::istringstream(value) >> *(property.ptr.v_double);
					break;
				case TYPE_STRING:
					*(property.ptr.v_string) = value;
					break;
				case TYPE_INTEGER_ARRAY:
				case TYPE_FLOAT_ARRAY:
				case TYPE_DOUBLE_ARRAY:
				case TYPE_UNKNOWN:
					result = INCOMPATIBLE_TYPES;
					break;
			}
		}
		data->host->sendError(result);
		return result;
	}

	ErrorCode Module::setProperty(const std::string &name, int *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

	ErrorCode Module::setProperty(const std::string &name, float *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

	ErrorCode Module::setProperty(const std::string &name, double *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

	int Module::getPropertyInt(const std::string &name, int element)
	{
		if(data->properties.count(name) > 0)
		{
			int value;
			ErrorCode error_status = data->properties[name].getValue(value, element);
			data->host->sendError(error_status);
			return value;
		}
		data->host->sendError(INVALID_PROPERTY);
		return 0;
	}

	int Module::getPropertyInt(int index, int element)
	{
		return getPropertyInt(getPropertyName(index), element);
	}

	float Module::getPropertyFloat(const std::string &name, int element)
	{
		if(data->properties.count(name) > 0)
		{
			float value;
			ErrorCode error_status = data->properties[name].getValue(value, element);
			data->host->sendError(error_status);
			return value;
		}
		data->host->sendError(INVALID_PROPERTY);
		return 0.0f;
	}

	float Module::getPropertyFloat(int index, int element)
	{
		return getPropertyFloat(getPropertyName(index), element);
	}

	double Module::getPropertyDouble(const std::string &name, int element)
	{
		if(data->properties.count(name) > 0)
		{
			double value;
			ErrorCode error_status = data->properties[name].getValue(value, element);
			data->host->sendError(error_status);
			return value;
		}
		data->host->sendError(INVALID_PROPERTY);
		return 0.0;
	}

	double Module::getPropertyDouble(int index, int element)
	{
		return getPropertyDouble(getPropertyName(index), element);
	}

	const std::string& Module::getPropertyString(const std::string &name, int element)
	{
		static std::string internal_bufferstring;
		internal_bufferstring = "";
		if(data->properties.count(name) > 0)
		{
			Property& property = data->properties[name];
			// check the type
			switch(property.type)
			{
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
				case TYPE_DOUBLE:
				{
					std::stringstream stream;
					stream <<  *(property.ptr.v_double);
					internal_bufferstring = stream.str();
					break;
				}
				case TYPE_STRING:
				{
					internal_bufferstring =  *(property.ptr.v_string);
					break;
				}
				case TYPE_INTEGER_ARRAY:
				{
					std::stringstream stream;
					stream << property.ptr.v_int[0];
					for(int i = 1; i < property.size; ++i)
					{
						stream << ", " << property.ptr.v_int[i];
					}
					internal_bufferstring = stream.str();
					break;
				}
				case TYPE_FLOAT_ARRAY:
				{
					std::stringstream stream;
					stream << property.ptr.v_float[0];
					for(int i = 1; i < property.size; ++i)
					{
						stream << ", " << property.ptr.v_float[i];
					}
					internal_bufferstring = stream.str();
					break;
				}
				case TYPE_DOUBLE_ARRAY:
				{
					std::stringstream stream;
					stream << property.ptr.v_double[0];
					for(int i = 1; i < property.size; ++i)
					{
						stream << ", " << property.ptr.v_double[i];
					}
					internal_bufferstring = stream.str();
					break;
				}
				default:
					data->host->sendError(INCOMPATIBLE_TYPES);
			}
		}		
		return internal_bufferstring;
	}

	const std::string& Module::getPropertyString(int index, int element)
	{
		return getPropertyString(getPropertyName(index), element);
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

	bool Module::hasProperty(const std::string &name) const
	{
		return data->properties.count(name) > 0;
	}

	const std::string& Module::getPropertyName(int index) const
	{
		static std::string dummystring = "";
		if((index < 0)||(index >= (int)(data->properties.size())))
		{
			data->host->sendError(INDEX_RANGE);
			return dummystring;
		}
		std::map<std::string, Property>::iterator itr = data->properties.begin();
		for(int i=0; i < index; i++)
			itr++;
		return itr->first;
	}

	PropertyType Module::getPropertyType(int index) const
	{
		static std::string dummystring = "";
		if((index < 0)||(index >= (int)(data->properties.size())))
		{
			data->host->sendError(INDEX_RANGE);
			return TYPE_UNKNOWN;
		}
		std::map<std::string, Property>::iterator itr = data->properties.begin();
		for(int i=0; i < index; i++)
			itr++;
		return itr->second.type;
	}

	PropertyType Module::getPropertyType(const std::string &name) const
	{
		if(data->properties.count(name) > 0)
			return data->properties[name].type;
		data->host->sendError(INVALID_PROPERTY);
		return TYPE_UNKNOWN;
	}

	int Module::getPropertySize(const std::string &name) const
	{
		if(data->properties.count(name) > 0)
			return data->properties[name].size;
		data->host->sendError(INVALID_PROPERTY);
		return 0;
	}

	int Module::getPropertySize(int index) const
	{
		return getPropertySize(getPropertyName(index));
	}

}
