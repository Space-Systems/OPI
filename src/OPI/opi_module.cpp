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
#include <fstream>
#include <iostream>
#include <algorithm>

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
				ErrorCode status = SUCCESS;
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
				ErrorCode status = SUCCESS;
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
				return SUCCESS;
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
            std::string configFileName;
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
        data->configFileName = "";
	}

	Module::~Module()
	{
	}

	ErrorCode Module::enable()
	{
		// already enabled, do nothing
		ErrorCode status = SUCCESS;
		if(!isEnabled())
			status = runEnable();
		if(status == SUCCESS)
            data->enabled = true;
        data->host->sendError(status);
		return status;
	}

	ErrorCode Module::disable()
	{
		ErrorCode status = SUCCESS;
		if(isEnabled())
		{
			status = runDisable();
			if(status == SUCCESS)
				data->enabled = false;
		}
		data->host->sendError(status);
		return status;
	}

	ErrorCode Module::runEnable()
	{
		return SUCCESS;
	}

	ErrorCode Module::runDisable()
	{
		return SUCCESS;
	}

	bool Module::isEnabled() const
	{
		return data->enabled;
	}

    void Module::setName(const char* newname)
	{
        data->name = std::string(newname);
	}

    const char* Module::getName() const
	{
        return data->name.c_str();
	}

    void Module::setAuthor(const char* author)
	{
        data->author = std::string(author);
	}

    const char* Module::getAuthor() const
	{
        return data->author.c_str();
	}

    void Module::setDescription(const char* description)
	{
        data->description = std::string(description);
	}

    const char* Module::getDescription() const
	{
        return data->description.c_str();
	}

	Host* Module::getHost() const
	{
		return data->host;
	}

    void Module::registerProperty(const char* name, int *location)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false)));
	}

    void Module::registerProperty(const char* name, float *location)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false)));
	}

    void Module::registerProperty(const char* name, double *location)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false)));
	}

    void Module::registerProperty(const char* name, std::string *location)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location)));
	}

    void Module::registerProperty(const char* name, int *location, int size)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false, size)));
	}

    void Module::registerProperty(const char* name, float *location, int size)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false, size)));
	}

    void Module::registerProperty(const char* name, double *location, int size)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(location, false, size)));
	}


    void Module::createProperty(const char* name, int value)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(&value, true)));
	}

    void Module::createProperty(const char* name, float value)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(&value, true)));
	}

    void Module::createProperty(const char* name, double value)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(&value, true)));
	}

    void Module::createProperty(const char* name, const char* value)
	{
        data->properties.insert(std::make_pair(std::string(name), Property(std::string(value))));
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
		return SUCCESS;
	}

    ErrorCode Module::setProperty(const char* name, int value)
	{
        return data->setValue(std::string(name), value);
	}
    ErrorCode Module::setProperty(const char* name, float value)
	{
        return data->setValue(std::string(name), value);
	}
    ErrorCode Module::setProperty(const char* name, double value)
	{
        return data->setValue(std::string(name), value);
	}

    ErrorCode Module::setProperty(const char* name, const char* value)
	{
		ErrorCode result = INVALID_PROPERTY;
		// find property
        if(data->properties.count(std::string(name)) > 0)
		{
			result = SUCCESS;
            Property& property = data->properties[std::string(name)];
			// check the type
			switch(property.type)
			{
				case TYPE_INTEGER:
                    std::istringstream(std::string(value)) >> *(property.ptr.v_int);
					break;
				case TYPE_FLOAT:
                    std::istringstream(std::string(value)) >> *(property.ptr.v_float);
					break;
				case TYPE_DOUBLE:
                    std::istringstream(std::string(value)) >> *(property.ptr.v_double);
					break;
				case TYPE_STRING:
                    *(property.ptr.v_string) = std::string(value);
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

    ErrorCode Module::setProperty(const char* name, int *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

    ErrorCode Module::setProperty(const char* name, float *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

    ErrorCode Module::setProperty(const char* name, double *value, int n)
	{
		ErrorCode result = INVALID_PROPERTY;
		data->host->sendError(result);
		return result;
	}

    int Module::getPropertyInt(const char* name, int element)
	{
        if(data->properties.count(std::string(name)) > 0)
		{
			int value;
            ErrorCode error_status = data->properties[std::string(name)].getValue(value, element);
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

    float Module::getPropertyFloat(const char* name, int element)
	{
        if(data->properties.count(std::string(name)) > 0)
		{
			float value;
            ErrorCode error_status = data->properties[std::string(name)].getValue(value, element);
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

    double Module::getPropertyDouble(const char* name, int element)
	{
        if(data->properties.count(std::string(name)) > 0)
		{
			double value;
            ErrorCode error_status = data->properties[std::string(name)].getValue(value, element);
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

    const char* Module::getPropertyString(const char* name, int element)
	{
		static std::string internal_bufferstring;
		internal_bufferstring = "";
        if(data->properties.count(std::string(name)) > 0)
		{
            Property& property = data->properties[std::string(name)];
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
        return internal_bufferstring.c_str();
	}

    const char* Module::getPropertyString(int index, int element)
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

    bool Module::hasProperty(const char* name) const
	{
        return data->properties.count(std::string(name)) > 0;
	}

    const char* Module::getPropertyName(int index) const
	{
		static std::string dummystring = "";
		if((index < 0)||(index >= (int)(data->properties.size())))
		{
			data->host->sendError(INDEX_RANGE);
            return dummystring.c_str();
		}
		std::map<std::string, Property>::iterator itr = data->properties.begin();
		for(int i=0; i < index; i++)
			itr++;
        return (itr->first).c_str();
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

    PropertyType Module::getPropertyType(const char* name) const
	{
        if(data->properties.count(std::string(name)) > 0)
            return data->properties[std::string(name)].type;
		data->host->sendError(INVALID_PROPERTY);
		return TYPE_UNKNOWN;
	}

    int Module::getPropertySize(const char* name) const
	{
        if(data->properties.count(std::string(name)) > 0)
            return data->properties[std::string(name)].size;
		data->host->sendError(INVALID_PROPERTY);
		return 0;
	}

	int Module::getPropertySize(int index) const
	{
		return getPropertySize(getPropertyName(index));
	}

    void Module::loadConfigFile()
    {
        loadConfigFile(data->configFileName.c_str());
    }

    void Module::loadConfigFile(const char* filename)
    {
        std::string filenameStr(filename);
        if (filenameStr != "" && filenameStr.length() > 4)
        {
            if (filenameStr.substr(filenameStr.length()-4,4) == ".cfg")
            {
                std::ifstream in(filename, std::ifstream::in);
                if (in.is_open())
                {
                    std::cout << "Applying settings for " << getName() << " from config file" << std::endl;
                    while (in.good())
                    {
                        std::string line;
                        std::getline(in, line);
                        line = trim(line);
                        if (line[0] != '#')
                        {
                            std::vector<std::string> setting = tokenize(line, "=");
                            if (setting.size() >= 2)
                            {
                                std::string property = trim(setting[0]);
                                std::string value = trim(setting[1]);

                                if (value.substr(0,1) == "\"" && value.substr(value.length()-1, value.length()) == "\"")
                                {
                                    if (!hasProperty(property.c_str()))
                                    {
                                        std::cout << "Registering new PropagatorProperty from config file: " << property << " (string)" << std::endl;
                                        createProperty(property.c_str(), value.substr(1,value.length()-2).c_str());
                                    }
                                    else setProperty(property.c_str(), value.substr(1,value.length()-2).c_str());
                                }
                                else if (value.find_first_of(".") != std::string::npos)
                                {
                                    if (value.substr(value.length()-1,1) == "f")
                                    {
                                        if (!hasProperty(property.c_str()))
                                        {
                                            std::cout << "Registering new PropagatorProperty from config file: " << property << " (float)" << std::endl;
                                            createProperty(property.c_str(), (float)atof(value.substr(0,value.length()-2).c_str()));
                                        }
                                        else setProperty(property.c_str(), (float)atof(value.substr(0,value.length()-2).c_str()));
                                    }
                                    else {
                                        if (!hasProperty(property.c_str()))
                                        {
                                            std::cout << "Registering new PropagatorProperty from config file: " << property << " (double)" << std::endl;
                                            createProperty(property.c_str(), atof(value.c_str()));
                                        }
                                        else setProperty(property.c_str(), atof(value.c_str()));
                                    }
                                }
                                else {
                                    if (!hasProperty(property.c_str()))
                                    {
                                        std::cout << "Registering new PropagatorProperty from config file: " << property << " (int)" << std::endl;
                                        createProperty(property.c_str(), atoi(value.c_str()));
                                    }
                                    else setProperty(property.c_str(), atoi(value.c_str()));
                                }
                            }
                        }
                    }
                    in.close();
                }
                else {
                    //std::cout << "No config file found for propagator " << getName() << std::endl;
                }
            }
            else {
                std::cout << filenameStr << " is not a valid config file for propagator " << getName() << std::endl;
            }
        }
        data->configFileName = filenameStr;
    }

    std::vector<std::string> Module::tokenize(std::string line, std::string delimiter)
    {
        std::vector<std::string> elements;

        std::string::size_type lastPos = line.find_first_not_of(delimiter, 0);
        std::string::size_type pos     = line.find_first_of(delimiter, lastPos);

        while (std::string::npos != pos || std::string::npos != lastPos)
        {
            elements.push_back(line.substr(lastPos, pos - lastPos));
            lastPos = line.find_first_not_of(delimiter, pos);
            pos = line.find_first_of(delimiter, lastPos);
        }
        return elements;
    }

    std::string Module::trim(const std::string &s)
    {
        auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c){return isspace(c); });
        auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c){return isspace(c); }).base();
        return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
    }

    int Module::requiresCUDA()
    {
        return 0;
    }

    int Module::requiresOpenCL()
    {
        return 0;
    }

    int Module::minimumOPIVersionRequired()
    {
        return 0;
    }

}
